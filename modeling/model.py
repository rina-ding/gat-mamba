from typing import Any, Dict, Optional
import torch
from torch.nn import (
    Linear,
    ReLU,
    Sequential,
)
from torch_geometric.nn import GATConv, global_mean_pool
import inspect
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch
from mamba_ssm import Mamba

def sinusoidal_positional_embedding(positions, embedding_dim, n=10000.0):

    if embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(embedding_dim))

    T = positions.shape[0]
    d = embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = positions.unsqueeze(-1).expand(-1, d//2)
    embeddings = torch.zeros((T, d), device=positions.device)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d).to(positions.device) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings

# Copied from https://github.com/bowang-lab/Graph-Mamba/blob/main/notebooks/mamba.ipynb
class GNNMambaLayer(torch.nn.Module):

    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        heads: int = 1,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        act: str = 'relu',
        att_type: str = 'transformer',
        d_state: int = 16,
        d_conv: int = 4,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.heads = heads
        self.dropout = dropout
        self.att_type = att_type
                
        if self.att_type == 'transformer':
            self.attn = torch.nn.MultiheadAttention(
                channels,
                heads,
                dropout=attn_dropout,
                batch_first=True,
            )
        if self.att_type == 'mamba':
            self.self_attn = Mamba(
                d_model=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=1
            )
            
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()
    
    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            # h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)

        ### Global attention transformer-style model.
        if self.att_type == 'transformer':
            h, mask = to_dense_batch(x, batch)
            h, _ = self.attn(h, h, h, key_padding_mask=~mask, need_weights=False)
            h = h[mask]
            
        if self.att_type == 'mamba':
            h, mask = to_dense_batch(x, batch)
            h = self.self_attn(h)[mask]
           
        h = F.dropout(h, p=self.dropout, training=self.training)
        # h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')

class GATMamba(torch.nn.Module):
    def __init__(self, uni_hidden, positional_embedding_size, gnn_dropout, mlp_dropout, num_heads, num_model_layers, model_type):
        super(GATMamba, self).__init__()
        self.num_heads_gnn = num_heads
        self.num_heads_transformer = 2
        self.sin_pe_dim = positional_embedding_size
        self.num_model_layers = num_model_layers
        self.num_uni_features = 1024
        self.num_continuous_edge_features = 2
        edge_embedding_size = 16
        self.num_luad_features = 512
        self.graph_dim_foundation = uni_hidden
        self.edge_embedding = torch.nn.Embedding(21, edge_embedding_size) # We have 21 subtype-subtype edge categories
        self.edge_linear_transform = Linear(self.num_continuous_edge_features, edge_embedding_size)
        self.uni_feature_linear_transform = Linear(self.num_uni_features, self.graph_dim_foundation)
        hidden = self.graph_dim_foundation + self.sin_pe_dim

        if num_model_layers == 1:
            self.conv1_gnn = GATConv(self.graph_dim_foundation + self.sin_pe_dim, hidden, heads = self.num_heads_gnn, dropout = gnn_dropout)
            self.conv1_all = GNNMambaLayer(hidden*self.num_heads_gnn, self.conv1_gnn, heads=self.num_heads_transformer, attn_dropout=mlp_dropout, dropout = mlp_dropout, att_type='mamba')
        else:
            self.conv1_gnn = GATConv(self.graph_dim_foundation + self.sin_pe_dim, hidden, heads = self.num_heads_gnn, dropout = gnn_dropout)
            self.conv1_all = GNNMambaLayer(hidden*self.num_heads_gnn, self.conv1_gnn, heads=self.num_heads_transformer, attn_dropout=mlp_dropout, dropout = mlp_dropout, att_type='mamba')

            self.conv2_gnn = GATConv(hidden*self.num_heads_gnn, hidden*self.num_heads_gnn, heads = self.num_heads_gnn, dropout = gnn_dropout)
            self.conv2_all = GNNMambaLayer(hidden*self.num_heads_gnn, self.conv2_gnn, heads=self.num_heads_transformer, attn_dropout=mlp_dropout, dropout = mlp_dropout, att_type='mamba')

        self.mlp = Sequential(
            Linear(hidden*self.num_heads_gnn, hidden*self.num_heads_gnn // 2),
            ReLU(),
            Linear(hidden*self.num_heads_gnn // 2, hidden*self.num_heads_gnn // 4),
            ReLU(),
            Dropout(mlp_dropout),
            Linear(hidden*self.num_heads_gnn // 4, 1),
)
    def forward(self, dataset):
    
        x, edge_index, edge_attr, batch = dataset.x, dataset.edge_index, dataset.edge_attr, dataset.batch
        x_index = 0
        y_index = 1
        feature_start_index = 2
       
        x_foundation = x[:, feature_start_index:feature_start_index+1024]
        x_foundation = self.uni_feature_linear_transform(x_foundation)

        positional_embedding = torch.cat([sinusoidal_positional_embedding(x[:,x_index], int(self.sin_pe_dim / 2)),
                                    sinusoidal_positional_embedding(x[:,y_index], int(self.sin_pe_dim / 2))], dim=-1)
        
        x = torch.cat([x_foundation, positional_embedding], dim = 1)
        categorical_embedding = self.edge_embedding(edge_attr[:, 0].view(-1, 1).to(dtype = torch.long)).squeeze(1)
        continuous_embedding = self.edge_linear_transform(edge_attr[:, 1:3])
        edge_attr = continuous_embedding + categorical_embedding
       
        if self.num_model_layers == 1:
            x = F.relu(self.conv1_all(x, edge_index, batch, edge_attr=edge_attr))
        else:
            x = F.relu(self.conv1_all(x, edge_index, batch, edge_attr=edge_attr))
            x = F.relu(self.conv2_all(x, edge_index, batch, edge_attr=edge_attr))

        x = global_mean_pool(x, batch)
        return self.mlp(x), x


    