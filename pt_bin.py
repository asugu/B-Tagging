from utils import *

if __name__ == '__main__':

    print_pt = True
    chk = "20_best"
    #chk = "18_MLP" 
    chk = "XGB"

    val_path = f'checkpoints/val_data/val_data_dict_{chk}.pkl'
    with open(val_path, 'rb') as file:
        val_data_dict = pickle.load(file)
    df_val = pd.DataFrame(val_data_dict)


    train_path = f'checkpoints/val_data/train_data_dict_{chk}.pkl'
    with open(train_path, 'rb') as file:
         train_data_dict = pickle.load(file)

    df_train = pd.DataFrame(train_data_dict)
    plot_pt_binned_eff(df_val , df_train, with_train=False, save_path=f"../plots/ptplot/ptbinning_{chk}")

