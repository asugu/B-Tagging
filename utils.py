from prettytable import PrettyTable
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import pandas as pd
from torchmetrics.classification import  BinaryROC
from scipy.interpolate import interp1d
import torch
import pickle

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def plot_metrics (x, train_loss, val_loss, train_accuracy, val_accuracy, train_auc, val_auc, train_f1, val_f1, save_path=None):
        
        
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharex=True, figsize=(20,5))
        clear_output(wait=True)
        
        #ax1.set_yscale('log')
        ax1.plot(x, train_loss, label="loss")
        ax1.plot(x, val_loss, label="val_loss")
        ax1.set_xlabel('epoch')
        ax1.legend()
        ax1.grid()
        
        ax2.plot(x, train_accuracy, label="accuracy({:.3f}%)".format(100*train_accuracy[-1]))
        max_acc = max(train_accuracy)
        ax2.plot(x, len(x)*[max_acc], 'b--', label="max acc. ({:.3f}%)".format(100*max_acc))
        ax2.plot(x, val_accuracy, label="val acc. ({:.3f}%)".format(100*val_accuracy[-1]))
        max_val_acc = max(val_accuracy)
        ax2.plot(x, len(x)*[max_val_acc], 'g--', label="max val. acc. ({:.3f}%)".format(100*max_val_acc))
        ax2.legend(loc="lower right")
        ax2.set_xlabel('epoch')
    #    ax2.set_ylim(0.5, 0.9)
        ax2.grid()
         
        ax3.plot(x, train_auc, label="AUC({:.3f})".format(train_auc[-1]))   #changed
        max_auc = max(train_auc)
        ax3.plot(x, len(x)*[max_auc], 'b--', label="max AUC ({:.3f})".format(max_auc))
        ax3.plot(x, val_auc, label="val AUC({:.3f})".format(val_auc[-1]))
        max_val_auc = max(val_auc)
        ax3.plot(x, len(x)*[max_val_auc], 'g--', label="max val. AUC ({:.3f})".format(max_val_auc))
        ax3.legend(loc="lower right")
        ax3.set_xlabel('epoch')
      #  ax3.set_ylim(0.5, 1.0)
       # ax3.yscale('log')
        ax3.grid()


        ax4.plot(x, train_f1, label="F1({:.3f})".format(train_f1[-1]))   #changed
        max_f1 = max(train_f1)
        ax4.plot(x, len(x)*[max_f1], 'b--', label="max F1 ({:.3f})".format(max_f1))
        ax4.plot(x, val_f1, label="val F1({:.3f})".format(val_f1[-1]))
        max_val_f1 = max(val_f1)
        ax4.plot(x, len(x)*[max_val_f1], 'g--', label="max val. F1 ({:.3f})".format(max_val_f1))
        ax4.legend(loc="lower right")
        ax4.set_xlabel('epoch')
       # ax4.set_ylim(0.0, 1.0)
        #ax3.scale('log')
        ax4.grid()
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()


def average_tprs_by_fprs(fprs, tprs):
    """
    For each unique FPR value, compute the average TPR value and optionally smooth the curve.

    Args:
    - fprs (array-like): False positive rates.
    - tprs (array-like): True positive rates.

    Returns:
    - smooth_fprs (array): Smoothed unique FPR values.
    - smooth_tprs (array): Interpolated TPR values for the smoothed FPRs.
    """
    fprs = np.array(fprs)
    tprs = np.array(tprs)

    # Find unique FPRs
    unique_fprs, indices = np.unique(fprs, return_inverse=True)
    avg_tprs = np.array([tprs[indices == i].mean() for i in range(len(unique_fprs))])

    # Interpolation for smoothing
    interp_func = interp1d(unique_fprs, avg_tprs, kind='linear', fill_value="extrapolate")
    smooth_fprs = np.linspace(0, 1, 100)  # Define resolution
    smooth_tprs = interp_func(smooth_fprs)

    return smooth_fprs, smooth_tprs


def remove_flav(preds, labels, flavors, n): 
   # Create a binary mask where 1 indicates the elements to remove
    mask = (flavors == n)
    
    preds = preds[mask != 1]
    labels = labels[mask != 1]
    
    return preds, labels

def calculate_rocs(preds, labels, flavors,remove=True):
    roc_metric = BinaryROC().to('cpu')

    preds = torch.tensor(preds).to('cpu').clone().detach()
    labels = torch.tensor(labels).to('cpu').clone().detach()

    labels = labels.int()

    if remove:
        c_preds, c_labels = remove_flav(preds, labels, flavors, 0)
        l_preds, l_labels = remove_flav(preds, labels, flavors, 4)

        c_fpr, c_tpr, _ = roc_metric(c_preds, c_labels)  
        l_fpr, l_tpr, _ = roc_metric(l_preds, l_labels) 

        return c_fpr.cpu().numpy(), c_tpr.cpu().numpy(), l_fpr.cpu().numpy(), l_tpr.cpu().numpy()

    l_preds, l_labels = remove_flav(preds, labels, flavors,4)

    l_fpr, l_tpr, _ = roc_metric(l_preds, l_labels) 

    return l_fpr.numpy(), l_tpr.numpy()

def get_eff_per_pt_bin(jet_pt, preds, labels, flavors, pt_min, pt_max, wp=0.1):

    jet_pt , preds, labels = np.array(jet_pt), np.array(preds), np.array(labels)

    preds = torch.tensor(preds).to('cpu')
    labels = torch.tensor(labels).to('cpu')
    flavors = torch.tensor(flavors).to('cpu')

    jet_pt = torch.tensor(jet_pt).to('cpu')

    idx = (jet_pt > pt_min) & (jet_pt < pt_max)
    # print(pt_min, pt_max)
    # print(len(idx))

    c_fpr, c_tpr, l_fpr, l_tpr = calculate_rocs(preds[idx], labels[idx], flavors[idx])

    c_interp = interp1d(c_fpr, c_tpr, bounds_error=False, fill_value=0)
    l_interp = interp1d(l_fpr, l_tpr, bounds_error=False, fill_value=0)

    c_eff = c_interp(wp)
    l_eff = l_interp(wp)

    return c_eff, l_eff

def plot_pt_binned_eff(df_val, df_train=None, with_train=False, save_path=None):
    """
    Plot efficiency vs pt for three working points.
    """

    jet_pt, preds, labels, flavors = df_val['pt'], df_val['preds'], df_val['labels'], df_val['flavors']
    if with_train == True:
        jet_pt_train, preds_train, labels_train, flavors_train = df_train['pt'], df_train['preds'], df_train['labels'], df_train['flavors']

    wps = [0.1, 0.01, 0.001]  # Working points (FPR)
    pts = [30, 100, 170, 240, 310, 380, 450, 520, 590, 660, 730, 800]  # pt bins with 70
    pts = [0, 70, 140, 210, 280, 350, 420, 490, 560, 630, 700, 770, 840]
   # pts = [30,150,250,350,450,550,650,750,850]
    effs_l = [[] for _ in range(len(wps))]
    effs_c = [[] for _ in range(len(wps))]

    if with_train == True:
        effs_l_train = [[] for _ in range(len(wps))]
        effs_c_train = [[] for _ in range(len(wps))]

    for i, wp in enumerate(wps):
        for j in range(len(pts) - 1):
            c_eff, l_eff = get_eff_per_pt_bin(jet_pt, preds, labels, flavors, pt_min=pts[j], pt_max=pts[j + 1], wp=wp)
            effs_l[i].append(l_eff)
            effs_c[i].append(c_eff)
            if with_train == True:
                c_eff_train, l_eff_train = get_eff_per_pt_bin(jet_pt_train, preds_train, labels_train, flavors_train, pt_min=pts[j], pt_max=pts[j + 1], wp=wp)
                effs_l_train[i].append(l_eff_train)
                effs_c_train[i].append(c_eff_train)

    # Plot the efficiencies
    colors = ['green', 'blue', 'red']
    labels = ['0.1 mistag', '0.01 mistag', '0.001 mistag']

    colors = ['green', 'blue', 'red']
    labels = ['0.1 mistag', '0.01 mistag', '0.001 mistag']

    pts = [pt+(pts[1]-pts[0])/2 for pt in pts]
    print(pts)

    plt.figure(figsize=(10, 6))
    for i in range(len(wps)):
        
        plt.errorbar(pts[:-1], effs_l[i], fmt='s', label=f"{labels[i]} Val", color=colors[i], linestyle='--', alpha=0.8)
        if with_train == True:
            plt.errorbar(pts[:-1], effs_l_train[i], fmt='s', label=f"{labels[i]} Train", color=colors[i], linestyle='dotted', alpha=0.8)

    plt.xlabel(r'$p_T$ [GeV]')
    plt.ylabel('b-jet Tagging Efficiency')
    plt.legend()
    plt.grid()
    #plt.xlim(0, 800)
    plt.ylim(0, 1)
    if save_path is not None:
        save_path_l = save_path + "_l.pdf"
        plt.savefig(save_path_l)
    
    else:
        plt.show()
    plt.close()

    plt.figure(figsize=(10, 6))
    for i in range(len(wps)):
        plt.errorbar(pts[:-1], effs_c[i], fmt='o', label=f"{labels[i]} Val", color=colors[i],linestyle='--', alpha=0.8)
        if with_train == True:
            plt.errorbar(pts[:-1], effs_c_train[i], fmt='s', label=f"{labels[i]} Train", color=colors[i], linestyle='dotted', alpha=0.8)

    plt.xlabel(r'$p_T$ [GeV]')
    plt.ylabel('b-jet Tagging Efficiency')
    #plt.title('b-jet Efficiency vs $p_T$')
    plt.legend()
    plt.grid()
    #plt.xlim(0, 800)
    plt.ylim(0, 1)

    if save_path is not None:
        save_path_c = save_path + "_c.pdf"
        plt.savefig(save_path_c)
    
    else:
        plt.show()
    plt.close()


def plot_ROC(c_fpr, c_tpr, l_fpr, l_tpr, save_path=None, benchmark=True):
    
    column_names = ['tpr', 'fpr']
    plt.figure()

    if benchmark==True:
        l_part_path = '/home/asugu/work/benchmark/l_part_roc.csv'
        l_deepjet_path = '/home/asugu/work/benchmark/l_deepjet_roc.csv'
        c_part_path = '/home/asugu/work/benchmark/c_part_roc.csv'
        c_deepjet_path = '/home/asugu/work/benchmark/c_deepjet_roc.csv'

        l_data_p = pd.read_csv(l_part_path, header=None, names=column_names)
        l_data_dj = pd.read_csv(l_deepjet_path, header=None, names=column_names)
        c_data_p = pd.read_csv(c_part_path, header=None, names=column_names)
        c_data_dj = pd.read_csv(c_deepjet_path, header=None, names=column_names)

        l_tpr_part = l_data_p['tpr']
        l_fpr_part = l_data_p['fpr']
        l_tpr_dj = l_data_dj['tpr']
        l_fpr_dj = l_data_dj['fpr']
        c_tpr_part = c_data_p['tpr']
        c_fpr_part = c_data_p['fpr']
        c_tpr_dj = c_data_dj['tpr']
        c_fpr_dj = c_data_dj['fpr']

        plt.plot(l_tpr_dj, l_fpr_dj, label='DeepJet - b vs. l', color='black')
        plt.plot(l_tpr_part, l_fpr_part, label='ParT - b vs. l', color='green')
        plt.plot(c_tpr_dj, c_fpr_dj, label='DeepJet - b vs. c', linestyle='--',color='black')
        plt.plot(c_tpr_part, c_fpr_part, label='ParT - b vs. c', linestyle='--',color='green')

    elif benchmark == False:
        val_path_MLP = f'checkpoints/val_data/val_data_dict_18_MLP.pkl'
        with open(val_path_MLP, 'rb') as file:
            val_data_dict_MLP = pickle.load(file)
        df_val_MLP = pd.DataFrame(val_data_dict_MLP)
        c_fpr_MLP, c_tpr_MLP, l_fpr_MLP, l_tpr_MLP = calculate_rocs(df_val_MLP['preds'], df_val_MLP['labels'], df_val_MLP['flavors'])
        plt.plot(l_tpr_MLP, l_fpr_MLP, label='MLP - b vs. l', color='green')
        plt.plot(c_tpr_MLP, c_fpr_MLP, label='MLP - b vs. c', linestyle='--',color='green')


        val_path_XGB = f'checkpoints/val_data/val_data_dict_XGB.pkl'
        with open(val_path_XGB, 'rb') as file:
            val_data_dict_XGB = pickle.load(file)
        df_val_XGB = pd.DataFrame(val_data_dict_XGB)
        c_fpr_XGB, c_tpr_XGB, l_fpr_XGB, l_tpr_XGB = calculate_rocs(df_val_XGB['probs'], df_val_XGB['labels'], df_val_XGB['flavors'])
        plt.plot(l_tpr_XGB, l_fpr_XGB, label='XGB - b vs. l', color='red')
        plt.plot(c_tpr_XGB, c_fpr_XGB, label='XGB - b vs. c', linestyle='--',color='red')


    plt.plot(c_tpr ,c_fpr,label='JetRetNet - b vs. c', linestyle='--',color='blue')
    plt.plot(l_tpr ,l_fpr,label='JetRetNet - b vs. l',color='blue')
    plt.ylabel('Misidentification Rate')
    plt.xlabel('Signal Efficiency')
    #plt.title('Mirrored ROC Curve')
    plt.yscale('log')
    plt.ylim(0.0009, 1.0)
    plt.legend()
    plt.grid(True, which="both", ls="--")

    if save_path is not None:
        plt.savefig(save_path)
    
    else:
        plt.show()
    plt.close()

if __name__ == '__main__':

    print_pt = True
    chk = "20_best"
    #chk = "18_MLP" 

    val_path = f'checkpoints/val_data/val_data_dict_{chk}.pkl'

    with open(val_path, 'rb') as file:
        val_data_dict = pickle.load(file)

    df_val = pd.DataFrame(val_data_dict)


    c_fpr, c_tpr, l_fpr, l_tpr = calculate_rocs(df_val['preds'], df_val['labels'], df_val['flavors'])

    print("after calculate rocs")

    plot_ROC(c_fpr, c_tpr, l_fpr, l_tpr, save_path=f"../plots/ROC/retnet_ROC_{chk}_XGB_benched.pdf", benchmark=False)

