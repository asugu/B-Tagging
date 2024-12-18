import torch
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
import argparse
#from torch.utils.data import DataLoader
from torch import nn
from torchmetrics.classification import BinaryAccuracy, AUROC, F1Score
from model_dataset import load_datasets, JetParticleDataset
import models
#from utils import count_parameters, plot_metrics, plot_ROC, calculate_rocs, average_tprs_by_fprs, plot_pt_binned_eff
from utils import *
import pickle
import time

std_pt = 58.70343399
mean_pt = 8.55453110e+01


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.003):
        """
        Args:
            patience (int): How many epochs to wait after the last improvement.
            min_delta (float): The minimum change in validation loss to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

def train_model(train_dataloader, test_dataloader, model, criterion, optimizer, n_epochs, device, learning_rate,patience=5):
    acc_metric = BinaryAccuracy().to(device)
    auc_metric = AUROC(task="binary").to(device)
    f1_metric = F1Score(task="binary").to(device)
    early_stopping = EarlyStopping(patience=patience)
    
    model.to(device)

    for i_epoch in range(n_epochs):
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
        train_losses, train_accuracies, train_aucs, train_f1s = [], [], [], []
        val_losses, val_accuracies, val_aucs, val_f1s = [], [], [], []

        lr = optimizer.param_groups[0]['lr']
        print(f'Learning rate set to {lr:.5f}.')

        model.train()
        train_data_dict = {'preds': [], 'pt': [], 'labels': [], 'flavors': []}
        for batch_inputs, batch_labels, flavors in tqdm(train_dataloader): 
            batch_track_inputs, batch_vertex_inputs, batch_jet_inputs = batch_inputs
            track_inputs = batch_track_inputs.transpose(1, 2)
            vertex_inputs = batch_vertex_inputs.transpose(1, 2)
            jet_inputs = batch_jet_inputs#.transpose(1, 2)
            inputs = (track_inputs, vertex_inputs, jet_inputs)

            train_label = batch_labels

            optimizer.zero_grad()
            train_pred = model(inputs).squeeze()

            loss = criterion(train_pred, train_label)
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()

            if i_epoch == (n_epochs-1):
                train_data_dict['preds'].extend(train_pred.detach().cpu().numpy())
                train_data_dict['pt'].extend((((inputs[2][:, 0].detach().cpu().numpy()) * std_pt) + mean_pt))
                train_data_dict['labels'].extend(train_label.detach().cpu().numpy())
                train_data_dict['flavors'].extend(flavors.detach().cpu().numpy())

            train_losses.append(loss.item())
            train_accuracy = acc_metric(train_pred.detach(), train_label.detach())
            train_accuracies.append(train_accuracy.item())
            train_auc = auc_metric(train_pred.detach(), train_label.detach())
            train_aucs.append(train_auc.item())
            train_f1 = f1_metric(train_pred.detach(), train_label.detach())
            train_f1s.append(train_f1.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_data_dict = {'preds': [], 'pt': [], 'labels': [], 'flavors': []}
            for val_inputs, val_labels, flavors in test_dataloader: 

                val_track_inputs, val_vertex_inputs, val_jet_inputs = val_inputs
                track_inputs = val_track_inputs.transpose(1, 2)
                vertex_inputs = val_vertex_inputs.transpose(1, 2)
                jet_inputs = val_jet_inputs
                val_inputs = (track_inputs, vertex_inputs, jet_inputs)

                # begin_time = time.time() 
                val_preds = model(val_inputs).squeeze()
                # end_time = time.time()
                # print("Inference time: ",(end_time - begin_time), len(val_inputs))

                val_loss = criterion(val_preds, val_labels)   
                val_losses.append(val_loss.item())
                val_accuracy = acc_metric(val_preds, val_labels)
                val_accuracies.append(val_accuracy.item())
                val_auc = auc_metric(val_preds, val_labels)
                val_aucs.append(val_auc.item())
                val_f1 = f1_metric(val_preds, val_labels)
                val_f1s.append(val_f1.item())

                val_data_dict['preds'].extend(val_preds.cpu().numpy())
                val_data_dict['pt'].extend((((val_inputs[2][:, 0].cpu().numpy()) * std_pt) + mean_pt))
                val_data_dict['labels'].extend(val_labels.cpu().numpy())
                val_data_dict['flavors'].extend(flavors.cpu().numpy())

            df = pd.DataFrame(val_data_dict)
            c_fpr, c_tpr, l_fpr, l_tpr = calculate_rocs(df['preds'], df['labels'], df['flavors'])

        epoch_val_loss.append(sum(val_losses) / len(val_losses))
        epoch_val_accuracy.append(sum(val_accuracies) / len(val_accuracies))
        epoch_val_auc.append(sum(val_aucs) / len(val_aucs))
        epoch_val_f1.append(sum(val_f1s) / len(val_f1s))

        epoch_train_loss.append(sum(train_losses) / len(train_losses))
        epoch_train_accuracy.append(sum(train_accuracies) / len(train_accuracies))
        epoch_train_auc.append(sum(train_aucs) / len(train_aucs))
        epoch_train_f1.append(sum(train_f1s) / len(train_f1s))

        plot_metrics(np.arange(i_epoch+epoch+1), epoch_train_loss, epoch_val_loss, epoch_train_accuracy, 
                     epoch_val_accuracy, epoch_train_auc, epoch_val_auc, 
                     epoch_train_f1, epoch_val_f1, save_path=f"../plots/perf/{args.model_type}_train.pdf")

        print(f'Training: Epoch [{i_epoch +epoch+ 1}/{n_epochs}] --- Loss: {epoch_train_loss[-1]:.4f} --- Accuracy: {epoch_train_accuracy[-1]:.4f} --- AUC: {epoch_train_auc[-1]:.3f}\n'
              f'Test: Epoch [{i_epoch+epoch + 1}/{n_epochs}] --- Loss: {epoch_val_loss[-1]:.4f} --- Accuracy: {epoch_val_accuracy[-1]:.4f} --- AUC: {epoch_val_auc[-1]:.3f}')

        plot_ROC(c_fpr, c_tpr, l_fpr, l_tpr, save_path=f"../plots/ROC/{args.model_type}_ROC.pdf")

        # Early Stopping check
        early_stopping(epoch_val_loss[-1])
        if early_stopping.early_stop:
            print(f"Early stopping after epoch {i_epoch+1}")
            break

        torch.cuda.empty_cache()
        del loss, val_loss, train_pred, val_preds
        del train_losses, train_accuracies, train_aucs, train_f1s
        del val_losses, val_accuracies, val_aucs, val_f1s
        gc.collect()
        
        if (i_epoch % 3 == 0) and (i_epoch != 0):
            checkpoint = {
                'epoch': i_epoch + epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': epoch_train_loss,
                'train_accuracies': epoch_train_accuracy,
                'train_aucs': epoch_train_auc,
                'train_f1s': epoch_train_f1,
                'val_losses': epoch_val_loss,
                'val_accuracies': epoch_val_accuracy,
                'val_aucs': epoch_val_auc,
                'val_f1s': epoch_val_f1,

            }
            checkpoint_path = f'checkpoints/{args.model_type}_model_checkpoint_{i_epoch+epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)

    print(i_epoch+epoch+1)
    checkpoint = {
        'epoch': i_epoch + epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': epoch_train_loss,
        'train_accuracies': epoch_train_accuracy,
        'train_aucs': epoch_train_auc,
        'train_f1s': epoch_train_f1,
        'val_losses': epoch_val_loss,
        'val_accuracies': epoch_val_accuracy,
        'val_aucs': epoch_val_auc,
        'val_f1s': epoch_val_f1,

    }

    checkpoint_path = f'checkpoints/model/{args.model_type}_model_checkpoint_final_{i_epoch+epoch+1}.pth'
    torch.save(checkpoint, checkpoint_path)

    with open(f'checkpoints/val_data/val_data_dict_{i_epoch+epoch+1}.pkl', 'wb') as f:
        pickle.dump(val_data_dict, f)
    
    with open(f'checkpoints/val_data/train_data_dict_{i_epoch+epoch+1}.pkl', 'wb') as f:
        pickle.dump(train_data_dict, f)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train a model for jet particle classification.')
    parser.add_argument('-ep', '--n_epochs', type=int, default=50, help='Number of epochs to train the model')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='Learning rate for the optimizer') # switchen from 0.0005 to 0.0001 on oct 21 17:00
    parser.add_argument('-bs', '--batch_size', type=int, default=2048, help='Batch size for training')
    parser.add_argument('-dv', '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to train the model on')
    parser.add_argument('-ly', '--layers', type=int, default=4, help='Number of layers in the model')
    parser.add_argument('-head', '--heads', type=int, default=8, help='Number of heads in the model')
    parser.add_argument('-hd', '--hidden_dim', type=int, default=32, help='Dimension of hidden layers')
    parser.add_argument('-ffn', '--ffn_size', type=int, default=64, help='Size of the feed-forward network')
    parser.add_argument('-typ', '--forward_type', type=str, choices=['parallel', 'recurrent', 'chunkwise'], default='parallel', help='Forward method type')
    parser.add_argument('-mdl', '--model_type', type=str, choices=['RetNet', 'MLP', 'LTC'], default='RetNet', help='model type')
    parser.add_argument('-chk', '--from_checkpoint', type=bool, default=False, help='Load model from checkpoint')
    parser.add_argument('-chk_pth', '--checkpoint_path', type=str, default='/home/asugu/work/B_Tagging/Scratch/Btag/RetNet/checkpoints/model_checkpoint_final_17_MLP.pth', help='Path to checkpoint file')

    args = parser.parse_args()

    print("Loading data...")
   # train_dataloader, test_dataloader = load_datasets(train_dataset_path='data/train_dataset_small.pt', test_dataset_path='data/test_dataset_small.pt', batch_size=args.batch_size)
    train_dataloader, test_dataloader = load_datasets(train_dataset_path='data/train_dataset_merged.pt', test_dataset_path='data/test_dataset_merged.pt', batch_size=args.batch_size)
    print("Constructing the model...")
    in_features = [10,9,6]  # The number of features of track, SV and jet properties

    if args.model_type == 'RetNet':
        model = models.RetNetModel(args.layers, args.hidden_dim, args.ffn_size, args.heads, input_dim=in_features, double_v_dim=True, forward_type=args.forward_type).to(args.device) # double_v_dim=False on oct 21
    elif args.model_type == 'MLP':
        model = models.MLP_Model(input_dim=211, hidden_layers=args.layers, nnodes=args.ffn_size).to(args.device)
    elif args.model_type == 'LTC':
        model = models.LTC_Model(input_dim=in_features, units=args.hidden_dim).to(args.device)
 
  #  model= nn.DataParallel(model,device_ids = [0, 1])
    
    criterion = nn.BCELoss().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Load from checkpoint if specified
    if args.from_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        epoch_train_loss = checkpoint['train_losses']
        epoch_train_accuracy = checkpoint['train_accuracies']
        epoch_train_auc = checkpoint['train_aucs']
        epoch_train_f1 = checkpoint['train_f1s']
        epoch_val_loss = checkpoint['val_losses']
        epoch_val_accuracy = checkpoint['val_accuracies']
        epoch_val_auc = checkpoint['val_aucs']
        epoch_val_f1 = checkpoint['val_f1s']
        scaler = True

        print(epoch_train_auc)

        # Ensure optimizer states are transferred to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
    else:
        epoch = 0 
        epoch_train_loss = []
        epoch_train_accuracy = []
        epoch_train_auc = []
        epoch_train_f1 = []
        epoch_val_loss = []
        epoch_val_accuracy = []
        epoch_val_auc = []
        epoch_val_f1 = []
        scaler = True

    model.to(args.device)
    count_parameters(model)

    # Train the model
    print("Training...")

    train_model(train_dataloader, test_dataloader, model, criterion, optimizer, args.n_epochs, args.device, args.learning_rate, patience=5)
