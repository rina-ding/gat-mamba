
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from glob import glob
import pandas as pd
from lifelines.utils import concordance_index
from sksurv.metrics import cumulative_dynamic_auc
from create_dataset_uni import GraphDataset
from torch_geometric.loader import DataLoader
from model import GATMamba
import matplotlib.pyplot as plt
import argparse

class TrainModel:
    def __init__(self, input_feature, num_classes, num_epcohs, batches, learning_rate, weight_decay, weight_regu, early_stopping_epoch, gnn_hidden,  positional_embedding_size, gnn_dropout, mlp_dropout, num_gat_heads, num_model_layers, model_type, fold_index):
        self.input_feature = input_feature
        self.num_classes = num_classes
        self.epochs = num_epcohs
        self.batch = batches
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight_regu = weight_regu
        self.early_stopping_epoch = early_stopping_epoch
        self.gnn_hidden = gnn_hidden
        self.positional_embedding_size = positional_embedding_size
        self.gnn_dropout = gnn_dropout
        self.mlp_dropout = mlp_dropout
        self.num_gat_heads = num_gat_heads
        self.num_model_layers = num_model_layers
        self.model_type = model_type
        self.fold_index = fold_index
        
        self.device = self._get_device()

    def _get_device(self):
        train_on_gpu = torch.cuda.is_available()
        if not train_on_gpu:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
        return device

    def CoxLoss(self, survtime, censor, hazard_pred, device):
        # This was copied from https://github.com/mahmoodlab/PathomicFusion
        current_batch_len = len(survtime)
        R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
        for i in range(current_batch_len):
            for j in range(current_batch_len):
                R_mat[i,j] = survtime[j] >= survtime[i]

        R_mat = torch.FloatTensor(R_mat).to(device)
        theta = hazard_pred.reshape(-1)
        exp_theta = torch.exp(theta)
        loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)
        return loss_cox
    
    def regularize_weights(self, model):
      # This was modified from https://github.com/mahmoodlab/PathomicFusion
        l1_reg = None

        for W in model.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
        return l1_reg

    def start_training(self, train_graph_dataset, val_graph_dataset):
        lambda_reg = self.weight_regu
       
        trainloader = DataLoader(train_graph_dataset, batch_size=self.batch, shuffle=True)
        validloader = DataLoader(val_graph_dataset, batch_size=self.batch, shuffle=True)

        model = GATMamba(self.gnn_hidden, self.positional_embedding_size, self.gnn_dropout, self.mlp_dropout, self.num_gat_heads, self.num_model_layers, self.model_type).to(self.device)
      
        optimizer = optim.Adam(model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.early_stopping_epoch, verbose=True)
        # Varibles to track
        train_losses, val_losses = [], []
        
        c_indices_list = []
        valid_loss_min = np.Inf
        tolerance_count = 0

        # Training loop
        epoch = 0
        while tolerance_count <= self.early_stopping_epoch and epoch <= self.epochs:
            print("-"*40)
            epoch += 1
            print('Epoch ', epoch)
            running_train_loss, running_val_loss = 0.0, 0.0
            epoch_loss = []
            # Put model on train mode
            model.train()
            for data in trainloader:
                graph_data = data.to(self.device)
                events = data.event.to(self.device)
                days = data.days.to(self.device)
                optimizer.zero_grad()
                output, _ = model(graph_data)
                output = output.squeeze(1)
                loss_reg = self.regularize_weights(model)
                loss = self.CoxLoss(days, events, output, self.device) + lambda_reg * loss_reg 

                loss.backward()
                optimizer.step()
                running_train_loss += float(loss.item()) * graph_data.num_graphs
                epoch_loss.append(float(loss.item() * graph_data.num_graphs))

            # Validation loop
            output_list = []
            event_list = []
            days_list = []
            with torch.no_grad():
                model.eval()
                for data in validloader:
                    graph_data = data.to(self.device)
                    events = data.event.to(self.device)
                    days = data.days.to(self.device)
                    output, _ = model(graph_data)
                    output = output.squeeze(1)
                    loss_reg = self.regularize_weights(model)
                    loss = self.CoxLoss(days, events, output, self.device) + lambda_reg * loss_reg 

                    running_val_loss += float(loss.item()) * graph_data.num_graphs
                    output_list.extend(-output.cpu().detach().numpy())
                    event_list.extend(events.cpu().detach().numpy())
                    days_list.extend(days.cpu().detach().numpy())

            # Calculate average losses
            avg_train_loss = running_train_loss / len(trainloader)
            avg_val_loss = running_val_loss / len(validloader)
            scheduler.step(avg_val_loss)
            
            c_index = concordance_index(days_list, output_list, event_list)

            # Append losses and track metrics
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            c_indices_list.append(c_index)

            # Print results
            print("Epoch:{}/{} - Training Loss:{:.6f} | Validation Loss: {:.6f}".format(
                epoch, self.epochs, avg_train_loss, avg_val_loss))
            print("C_index:{}".format(
                c_index))

            # Save model
            if avg_val_loss <= valid_loss_min:
                print("Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(valid_loss_min, avg_val_loss))
                print("-" * 40)
               
                model_to_save = model
                torch.save(model_to_save.state_dict(), os.path.join(MODEL_DIR, 'model_fold' + str(self.fold_index) + '.pth'))
                # Update minimum loss
                valid_loss_min = avg_val_loss
                tolerance_count = 0
            elif avg_val_loss > valid_loss_min:
                tolerance_count += 1

            # Save plots
            figures_dir = os.path.join(FIGURE_DIR, 'fold' + str(self.fold_index))
            if not os.path.exists(figures_dir):
                os.makedirs(figures_dir)
            plt.plot(train_losses, label='Training loss')
            plt.plot(val_losses, label='Validation loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(figures_dir, 'losses.png'))
            plt.clf()

            plt.plot(c_indices_list, label = 'Validation C-index')        
            plt.xlabel('Epoch')
            plt.ylabel('Score')
            plt.legend()
            plt.savefig(os.path.join(figures_dir, 'val_c_index.png'))
            plt.clf()

    def runinference(self, test_graph_dataset):
       
        figures_dir = os.path.join(FIGURE_DIR, 'fold' + str(self.fold_index))
        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)
       
        print("Graphs for testing:", len(test_graph_dataset))
        testloader = DataLoader(test_graph_dataset, batch_size=self.batch, shuffle=False)
      
        model = GATMamba(self.gnn_hidden, self.positional_embedding_size, self.gnn_dropout, self.mlp_dropout, self.num_gat_heads, self.num_model_layers, self.model_type).to(self.device)

        print("Model Weights Loaded")
        weights = torch.load(os.path.join(MODEL_DIR, 'model_fold' + str(self.fold_index) + '.pth'))
        model.load_state_dict(weights)
        model.to(self.device)

        # Make Predictions
        output_list = []
        output_list_raw = []

        event_list = []
        days_list = []
        pid_list = []
        with torch.no_grad():
            model.eval()
            for data in testloader:
                graph_data = data.to(self.device)
                events = data.event.to(self.device)
                days = data.days.to(self.device)
                pid = data.pid
                output, _ = model(graph_data)
                output_list_raw.extend(output.cpu().detach().numpy())
                output_list.extend(-output.cpu().detach().numpy())
                event_list.extend(events.cpu().detach().numpy())
                days_list.extend(days.cpu().detach().numpy())
                pid_list.extend(pid)

            c_index = concordance_index(days_list, output_list, event_list)
            data = {'pid':pid_list,
                'risk': output_list_raw, 
                    'event': event_list,
                    'days': days_list,
                    }
            df_save_prediction = pd.DataFrame(data)
            df_save_prediction['risk'] = df_save_prediction['risk'].str[0]

        return c_index, df_save_prediction

def get_auc_score(df_train, df_test):
    dtype = np.dtype([('event', np.bool_), ('time', np.float64)])
    structured_array_train = np.array(list(df_train[['event', 'days']].to_records(index=False)), dtype=dtype)
    structured_array_test = np.array(list(df_test[['event', 'days']].to_records(index=False)), dtype=dtype)
    _, score = cumulative_dynamic_auc(structured_array_train, structured_array_test, df_test['risk'], np.asarray([365, 365*3, 365*5]).flatten()) 
    return score

def split_graph_data(df_clinical, dataset):
    split_indices = []
    for i in range(len(dataset)):
        for j in range(len(df_clinical)):
            if dataset[i].pid == df_clinical['pid'][j]:
                split_indices.append(i)
    splitted_dataset = dataset[split_indices]
    
    return splitted_dataset

if __name__ == "__main__":
    torch.manual_seed(123) 
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_data_path', type = str, default = None, help = 'path to the saved graph dataset file in .pt format')
    parser.add_argument('--cv_split_path', type = str, default = None, help = 'path to 5-fold cv train, validation, and test cases')
    parser.add_argument('--num_epochs', type = int, default = 200, help = 'the maximum number of training epochs')
    parser.add_argument('--batches', type = int, default = 16, help = 'batch size')
    parser.add_argument('--early_stopping_epoch', type = int, default = 5, help = 'number of epochs for early stopping in training')
    parser.add_argument('--num_model_layers', type = int, default = 1, help = 'number of GATMambaBlocks (or number of layers)')
    parser.add_argument('--learning_rate', type = float, default = 5e-5, help = 'learning rate')
    parser.add_argument('--weight_decay', type = float, default = 1e-4, help = 'weight decay')
    parser.add_argument('--weight_regu', type = float, default = 1e-5, help = 'weight regularization factor')
    parser.add_argument('--gnn_hidden', type = int, default = 64, help = 'the GAT-Mamba UNI node feature hidden dimension')
    parser.add_argument('--positional_embedding_size', type = int, default = 16, help = 'positional embedding dimension')
    parser.add_argument('--gnn_dropout', type = float, default = 0.1, help = 'dropout for GAT conv layer')
    parser.add_argument('--mlp_dropout', type = float, default = 0.3, help = 'dropout for the MLP layer')
    parser.add_argument('--num_gat_heads', type = int, default = 1, help = 'number of attention heads in GAT')

    args = parser.parse_args()

    input_feature = 'uni_features' 
    model_type = 'gat_mamba'

    num_epcohs = args.num_epochs
    batches = args.batches
    early_stopping_epoch = args.early_stopping_epoch
    num_model_layers = args.num_model_layers
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    weight_regu = args.weight_regu
    gnn_hidden = args.gnn_hidden
    positional_embedding_size = args.positional_embedding_size
    gnn_dropout = args.gnn_dropout
    mlp_dropout = args.mlp_dropout
    num_gat_heads = args.num_gat_heads

    num_classes = 1

    FIGURE_DIR = './figures'
    MODEL_DIR = './saved_models'
    if not os.path.exists(FIGURE_DIR):
        os.makedirs(FIGURE_DIR)
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    train_scores = []
    test_scores = []
    val_scores = []
    test_scores_auc = []
 
    processed_data_path = args.graph_data_path
    cv_split_path = args.cv_split_path
    graph_dataset = GraphDataset(root = processed_data_path)

    for fold_index in range(5):
        df_train_path = os.path.join(cv_split_path, 'fold' + str(fold_index), 'pids_train.csv')
        df_val_path = os.path.join(cv_split_path, 'fold' + str(fold_index), 'pids_val.csv')
        df_test_path = os.path.join(cv_split_path, 'fold' + str(fold_index), 'pids_test.csv')
        df_train = pd.read_csv(df_train_path)
        df_val = pd.read_csv(df_val_path)
        df_test = pd.read_csv(df_test_path)
        train_graph_dataset = split_graph_data(df_train, graph_dataset)
        val_graph_dataset = split_graph_data(df_val, graph_dataset)
        test_graph_dataset = split_graph_data(df_test, graph_dataset)

        train_obj = TrainModel(input_feature, num_classes, num_epcohs, batches, learning_rate, weight_decay, weight_regu, early_stopping_epoch, gnn_hidden,  positional_embedding_size, gnn_dropout, mlp_dropout, num_gat_heads, num_model_layers, model_type, fold_index)
        train_obj.start_training(train_graph_dataset, val_graph_dataset)
        train_score, df_prediction_train = train_obj.runinference(train_graph_dataset)
        train_scores.append(train_score)
        val_score, df_prediction_val = train_obj.runinference(val_graph_dataset)
        val_scores.append(val_score)
        test_score, df_prediction_test = train_obj.runinference(test_graph_dataset)
        test_scores.append(test_score)

        df_prediction_train.to_csv(os.path.join(FIGURE_DIR, 'fold' + str(fold_index) + 'prediction_train.csv'), index = False)
        df_prediction_test.to_csv(os.path.join(FIGURE_DIR, 'fold' + str(fold_index) + 'prediction_test.csv'), index = False)

        auc_score = get_auc_score(df_prediction_train, df_prediction_test)
        test_scores_auc.append(auc_score)

        fold_index += 1

    train_scores = [round(i, 3) for i in train_scores]
    train_scores_array = np.asarray(train_scores)
    print('Train results: ')
    print(train_scores)
    print('Average train scores ', np.mean(train_scores_array, axis = 0))
    print('SD train scores ', np.std(train_scores_array, axis = 0))
    print('----------------')

    val_scores = [round(i, 3) for i in val_scores]
    val_scores_array = np.asarray(val_scores)
    print('Validation results: ')
    print(val_scores)
    print('Average val scores ', np.mean(val_scores_array, axis = 0))
    print('SD val scores ', np.std(val_scores_array, axis = 0))
    print('----------------')

    test_scores = [round(i, 3) for i in test_scores]
    test_scores_array = np.asarray(test_scores)
    print('Testing results: ')
    print(test_scores)
    print('Average test scores ', np.mean(test_scores_array, axis = 0))
    print('SD test scores ', np.std(test_scores_array, axis = 0))

    test_scores_auc = [round(i, 3) for i in test_scores_auc]
    test_scores_array_auc = np.asarray(test_scores_auc)
    print('Testing results AUC : ')
    print(test_scores_auc)
    print('Average test scores AUC', np.mean(test_scores_array_auc, axis = 0))
    print('SD test scores AUC', np.std(test_scores_array_auc, axis = 0))


   