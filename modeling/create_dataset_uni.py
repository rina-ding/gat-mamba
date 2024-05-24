import numpy as np
import pandas as pd
from glob import glob
import os
from sklearn.neighbors import kneighbors_graph
from natsort import natsorted
from sklearn.metrics.pairwise import euclidean_distances
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch
from torch_geometric.utils import to_undirected

class GraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return 'graph_data.pt'

    def process(self):
        all_dir = natsorted(glob(os.path.join('parent dir to the extracted tile/node features', '*.csv')))
        data_list = create_data_object_list(all_dir)

        # Store the processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])  

def build_graph_as_data_object(df_slide, k_in_knn, pid):
    
    edge_index_raw, xy_matrix = get_edge_index(df_slide,  k_in_knn)
    df_slide.insert(df_slide.columns.get_loc('predicted_class')+1, 'tile_x', xy_matrix[:, 0])
    df_slide.insert(df_slide.columns.get_loc('predicted_class')+2, 'tile_y', xy_matrix[:, 1])
   
    df_edge_features = pd.DataFrame(columns = ['node1', 'node2', 'luad_subtype_edge', 'node_distance_eucl', 'euclidean_distance_uni'])

    for i in range(len(edge_index_raw[0])):
        start_node_index = edge_index_raw[0][i]
        end_node_index = edge_index_raw[1][i]
        if 'TCGA' in df_slide['tile_name'][start_node_index]:
            start_node_x = int(df_slide['tile_name'][start_node_index].split('-')[9].replace('x', ''))
            start_node_y = int(df_slide['tile_name'][start_node_index].split('-')[10].replace('y', ''))
            
            end_node_x = int(df_slide['tile_name'][end_node_index].split('-')[9].replace('x', '')) 
            end_node_y = int(df_slide['tile_name'][end_node_index].split('-')[10].replace('y', '')) 
        else:
            start_node_x = int(df_slide['tile_name'][start_node_index].split('-')[4].replace('x', ''))
            start_node_y = int(df_slide['tile_name'][start_node_index].split('-')[5].replace('y', ''))
            
            end_node_x = int(df_slide['tile_name'][end_node_index].split('-')[4].replace('x', '')) 
            end_node_y = int(df_slide['tile_name'][end_node_index].split('-')[5].replace('y', '')) 

        df_edge_features.loc[i, 'node1'] = df_slide['tile_name'][start_node_index]
        df_edge_features.loc[i, 'node2'] = df_slide['tile_name'][end_node_index]

        # Subtype-subtype edge
        list_of_subtypes = [df_slide['predicted_class'][start_node_index], df_slide['predicted_class'][end_node_index]]
        df_edge_features.loc[i, 'luad_subtype_edge'] = subtype_subtype_edge(list_of_subtypes)
        if df_edge_features['luad_subtype_edge'][i] == 'none':
            list_of_subtypes_reversed = [df_slide['predicted_class'][end_node_index], df_slide['predicted_class'][start_node_index]]
            df_edge_features.loc[i, 'luad_subtype_edge'] = subtype_subtype_edge(list_of_subtypes_reversed)

        # Node Euclidean distance
        node_distance_eucl = euclidean_distances([[start_node_x, start_node_y], [end_node_x, end_node_y]])[0][1]
        df_edge_features.loc[i, 'node_distance_eucl'] = node_distance_eucl

        # Euclidean distance of UNI features
        start_idx_uni = df_slide.columns.get_loc('tile_y') + 1
        end_idx_uni = start_idx_uni + 1024
        node1_uni_features = np.reshape(df_slide.iloc[start_node_index, start_idx_uni:end_idx_uni].values, (1, -1))
        node2_uni_features = np.reshape(df_slide.iloc[end_node_index, start_idx_uni:end_idx_uni].values, (1, -1))
        euclidean_distance_uni = euclidean_distances(node1_uni_features, node2_uni_features)
        df_edge_features.loc[i, 'euclidean_distance_uni'] = float(euclidean_distance_uni)

    node_features_start_idx = df_slide.columns.get_loc('predicted_class') + 1 
    node_features = torch.tensor(df_slide.iloc[:, node_features_start_idx:].values, dtype=torch.float)
    edge_features = torch.tensor(df_edge_features.iloc[:, df_edge_features.columns.get_loc('node2') + 1:].values.astype(float), dtype = torch.float)
    result_undirected = to_undirected(edge_index = torch.tensor(edge_index_raw, dtype=torch.long), edge_attr = edge_features, reduce = 'mean')
    edge_index_undirected = result_undirected[0]
    edge_features_undirected = result_undirected[1]

    data = Data(x = node_features, edge_index = edge_index_undirected, edge_attr = edge_features_undirected)
    data.event = torch.tensor(df_slide['event'][0], dtype=torch.float)
    data.days = torch.tensor(df_slide['days'][0], dtype=torch.float)
    data.pid = pid
    data.sid = df_slide['sid'][0]

    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')

    return data

def create_data_object_list(node_features_dir):
    
    k_in_knn = 8

    data_list = []
    for patient in range(len(node_features_dir)):
        pid = str(os.path.basename(node_features_dir[patient].replace('.csv', '')))
        print(pid)
        df = pd.read_csv(node_features_dir[patient])
        df['sid'] = df['sid'].astype(str)

        # Need to generate graph data for each slide separately if there are multiple slides per patient
        groups_obj = df.groupby('sid')
        this_patient_data_list = []
        for group_name, df_slide in groups_obj:
            df_slide_original = df_slide.reset_index(drop = True)
            print('Slide ID ', group_name)
            print('length ', len(df_slide_original))

            if len(df_slide_original) >= 10:
                data = build_graph_as_data_object(df_slide_original, k_in_knn, pid)
                this_patient_data_list.append(data)
        
        temp = Batch.from_data_list(this_patient_data_list)
        this_patient_data = Data(x = temp.x, edge_index = temp.edge_index, edge_attr = temp.edge_attr)
        this_patient_data.event = torch.tensor(df_slide_original['event'][0], dtype=torch.float)
        this_patient_data.days = torch.tensor(df_slide_original['days'][0], dtype=torch.float)
        this_patient_data.pid = pid

        data_list.append(this_patient_data)

    return data_list

def get_edge_index(df,  k_in_knn):
    # Get the x, y coordinates of the tiles
    xy_matrix = np.empty((len(df), 2))
    for tile in range(len(df)):
        tile_name = df['tile_name'][tile]
        if 'TCGA' not in tile_name:
            x = int(tile_name.split('-')[4].replace('x', '')) 
            y = int(tile_name.split('-')[5].replace('y', '')) 
        else:
            x = int(tile_name.split('-')[9].replace('x', '')) 
            y = int(tile_name.split('-')[10].replace('y', '')) 
        xy_matrix[tile][0] = x
        xy_matrix[tile][1] = y

    A = kneighbors_graph(xy_matrix,  k_in_knn, mode='connectivity', include_self=False)
    A = A.tocoo()
    row = A.row
    col = A.col    
    edge_index = np.stack([row, col])

    return edge_index, xy_matrix

def subtype_subtype_edge(list_of_subtypes):
        if list_of_subtypes == [0, 0]:
            return 0
        elif list_of_subtypes == [1, 0]:
            return 1
        elif list_of_subtypes == [2, 0]:
            return 2
        elif list_of_subtypes == [3, 0]:
            return 3
        elif list_of_subtypes == [4, 0]:
            return 4
        elif list_of_subtypes == [5, 0]:
            return 5
        elif list_of_subtypes == [1, 1]:
            return 6
        elif list_of_subtypes == [2, 1]:
            return 7
        elif list_of_subtypes == [3, 1]:
            return 8
        elif list_of_subtypes == [4, 1]:
            return 9
        elif list_of_subtypes == [5, 1]:
            return 10
        elif list_of_subtypes == [2, 2]:
            return 11
        elif list_of_subtypes == [3, 2]:
            return 12
        elif list_of_subtypes == [4, 2]:
            return 13
        elif list_of_subtypes == [5, 2]:
            return 14
        elif list_of_subtypes == [3, 3]:
            return 15
        elif list_of_subtypes == [4, 3]:
            return 16
        elif list_of_subtypes ==[5, 3]:
            return 17
        elif list_of_subtypes ==[4, 4]:
            return 18
        elif list_of_subtypes ==[5, 4]:
            return 19
        elif list_of_subtypes ==[5, 5]:
            return 20
        else:
            return 'none'

if __name__ == "__main__":

    processed_data_save_path = ''
    if not os.path.exists(processed_data_save_path):
        os.makedirs(processed_data_save_path)
    dataset = GraphDataset(root = processed_data_save_path)
    
            

            