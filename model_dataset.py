import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
import gc

global_jet_mean = np.zeros(6)
global_jet_std = np.zeros(6)
global_track_mean = np.zeros(10)
global_track_std = np.zeros(10)
global_sv_mean = np.zeros(9)
global_sv_std = np.zeros(9)

class JetParticleDataset(torch.utils.data.Dataset):
    def __init__(self, df, device='cpu', eval=False,transform=None, scaler=False):
        self.df = df
        self.transform = transform
        self.device = device

        self.jet_features = df[['jet_mass', 'jet_pt', 'jet_eta', 'jet_phi', 'jet_track_count', 'jet_sv_count']]
        
        self.particle_features = df[['track_E', 'track_pt', 'track_charge','track_pid', 'track_d0', 'track_dz', 'track_d0_sig', 'track_dz_sig', 'track_deta', 'track_dphi']]
        self.vertex_features = df[['sv_pt', 'sv_ntracks', 'sv_mass', 'sv_chi2', 'sv_ndof', 'sv_dxy', 'sv_dlen', 'sv_dxy_sig', 'sv_dlen_sig']]

        self.unpad_track_len = df['jet_track_count']
        self.unpad_sv_len = df['jet_sv_count']

        self.labels = df['jet_btag']
        self.flavor = df['hadron_flav'] 

        self.scaler = scaler  
        self.eval = eval   

        if self.scaler:
            self.StandardScaler_()

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):

        jet_inputs = torch.tensor(self.jet_features.iloc[idx].values, dtype=torch.float32, device=self.device)

        particle_inputs = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.particle_features.iloc[idx].values if isinstance(arr, np.ndarray)]
        vertex_inputs = [torch.tensor(arr, dtype=torch.float32, device=self.device) for arr in self.vertex_features.iloc[idx].values if isinstance(arr, np.ndarray)]

        label = torch.tensor(self.labels.iloc[idx], dtype=torch.float32,device=self.device)
        flavor = torch.tensor(self.flavor.iloc[idx], dtype=torch.float32,device=self.device)

        particle_inputs = torch.stack(particle_inputs)
        vertex_inputs = torch.stack(vertex_inputs)
          
        if self.transform is not None:
            particle_inputs = self.transform(particle_inputs)
            vertex_inputs = self.transform(vertex_inputs)
            jet_inputs = self.transform(jet_inputs)

        combined_inputs = (particle_inputs, vertex_inputs, jet_inputs)
        return combined_inputs, label, flavor

    def StandardScaler_(self):

        if not self.eval:

            i = 0
            for col in self.jet_features.columns:
                print(f"Calculating scaling for jet feature: {col} ...")
                global_jet_mean[i] = np.mean(self.jet_features[col])
                global_jet_std[i] = np.std(self.jet_features[col])
                i += 1
            print("Global jet mean: ", global_jet_mean)
            print("Global jet std: ", global_jet_std)

            i = 0
            for col in self.particle_features.columns:
                print(f"calculating {col} ...")
                mean_arr = []
                var_c = 0    # combined variation
                len_arr = 0 
                for arr, lenght in zip(self.particle_features[col],self.unpad_track_len):
                    mean_arr.append(np.mean(arr[:lenght]))
                    if len_arr == 0:
                        sum = np.var(arr[:lenght])
                    else:
                        sum = (var_c / len_arr) + (np.var(arr) / len(arr))
                    len_arr += len(arr[:lenght])
                    var_c = sum
                global_track_mean[i] = np.mean(mean_arr)
                global_track_std[i] = np.sqrt(var_c)
                i += 1
            print ("mean track array is: ",global_track_mean)
            print ("std track array is: ",global_track_std)


            i = 0
            for col in self.vertex_features.columns:
                print(f"calculating {col} ...")
                mean_arr = []
                var_c = 0    # combined variation
                len_arr = 0 
                for arr, lenght in zip(self.vertex_features[col],self.unpad_sv_len):
                    mean_arr.append(np.mean(arr[:lenght]))
                    if len_arr == 0:
                        sum = np.var(arr[:lenght])
                    else:
                        sum = (var_c / len_arr) + (np.var(arr) / len(arr))
                    len_arr += len(arr[:lenght])
                    var_c = sum
                global_sv_mean[i] = np.mean(mean_arr)
                global_sv_std[i] = np.sqrt(var_c)
                i += 1
            print ("mean sv array is: ",global_sv_mean)
            print ("std sv array is: ",global_sv_std)


        ##### Calculated for 2.5M jet '../data/event_data_sv_pad16_33.pkl' #####
        # global_jet_mean = np.array([ 1.22791615e+01, 8.55609512e+01, -5.19885682e-04, -5.31661464e-03, 8.04646430e+00, 1.23135133e+00])
        # global_jet_std = np.array([7.56288385, 58.72772598, 1.17901385, 1.8082757, 3.59384253, 0.51541153])
     
        # global_track_mean = np.array([1.21683559e+01, 6.58074760e+00, 8.69581848e-03, 1.88703299e+00, -7.38178054e-03, -4.06104140e-03, 3.51619768e+00, 1.00649805e+01, 7.93984873e-06, 4.87137208e-04])
        # global_track_std = np.array([2.62111205e+00, 1.82100381e+00, 1.58576433e-01, 2.93560074e+01, 1.81225960e-01, 2.73429730e-02, 1.79218822e+00, 2.57825534e-01, 1.49157322e-02, 9.19927578e-03])
     
        # global_sv_mean = np.array([23.46401596,  3.23178649, 1.59123814, 3.10244203, 3.20518279, 0.68311387, 1.21625102, 16.97854424, 17.01024246])
        # global_sv_std = np.array([4.03626142, 0.35777098, 0.10691383, 0.2646511, 0.15863598, 0.2965479, 0.45432767, 2.39560448, 2.36696998])
        

        ##### Calculated for 5M jet '../data/event_data_sv_pad(5_16)_merged.pkl' #####
        # global_jet_mean = np.array([ 1.22772627e+01, 8.55453110e+01, 2.42096372e-03, -5.83897391e-03, 8.05747473e+00, 1.23182168e+00])
        # global_jet_std = np.array([ 7.55819845, 58.70343399, 1.17017901, 1.80904961, 3.59147004, 0.51563964])
     
        # global_track_mean = np.array([ 1.20785046e+01, 6.58131933e+00, 8.61030724e-03, 1.85943472e+00, -7.30540184e-03, -4.31924593e-03, 3.52442551e+00, 1.00410547e+01, -4.97698202e-05, 4.92945081e-04])
        # global_track_std = np.array([6.59886849e-01, 5.42531279e-01, 1.76776697e-01, 3.72998830e+01, 8.93101812e-02, 6.27518111e-01, 7.29065021e-01, 4.82284526e+01, 2.10426883e-02, 1.09323385e-02])

        i = 0
        print("applying scaling for jet features")
        print(self.eval, global_jet_mean)
        for col in self.jet_features.columns:
            self.jet_features.loc[:, col] = (self.jet_features[col] - global_jet_mean[i]) / global_jet_std[i]
            i += 1

        i = 0
        print("applying scaling for track features")
        print(self.eval, global_track_mean)
        for col in self.particle_features:
            for arr, lenght in zip(self.particle_features[col],self.unpad_track_len):
                arr[:lenght] = (arr[:lenght] - global_track_mean[i]) / global_track_std[i]
            i += 1

        i = 0
        print("applying scaling for vertex features")
        print(self.eval, global_sv_mean)
        for col in self.vertex_features:
            for arr, lenght in zip(self.vertex_features[col],self.unpad_sv_len):
                arr[:lenght] = (arr[:lenght] - global_sv_mean[i]) / global_sv_std[i]
            i += 1

    def MinMaxScaler_(self):
        if not self.eval:
            i = 0
            for col in self.particle_features.columns:
                min_arr = []
                max_arr = []
                for arr, lenght in zip(self.particle_features[col],self.unpad_track_len):
                    min_arr.append(np.min(arr[:lenght]))
                    max_arr.append(np.max(arr[:lenght]))
                global_min[i] = np.min(min_arr)
                global_max[i] = np.max(max_arr)
                i += 1
            print ("min array is: ",global_min)
            print ("max array is: ",global_max)

        i = 0
        for col in self.particle_features:
            diff = global_max[i] - global_min[i]
            print(f"The difference for {col} is {diff}")
            for arr, lenght in zip(self.particle_features[col],self.unpad_track_len):
                arr[:lenght] = (arr[:lenght] - global_min[i])/diff
            i += 1


def load_datasets(train_dataset_path='train_dataset.pt', test_dataset_path='test_dataset.pt', batch_size=32):

    train_dataset = torch.load(train_dataset_path)
    print(f'Train dataset size: {len(train_dataset)}')
    test_dataset = torch.load(test_dataset_path)
    print(f'Test dataset size: {len(test_dataset)}')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=int(len(test_dataset)/4), shuffle=True, drop_last=True)

    print(f'Test DataLoader size: {len(test_dataloader)}')

    del train_dataset, test_dataset
    gc.collect()

    return train_dataloader, test_dataloader

def main():
    file_path = f'../data/event_data_sv_pad(5_16)_merged.pkl' #5 M
    #file_path = f'../data/event_data_sv_pad16.pkl'
    with open(file_path, 'rb') as file:
        event_data = pickle.load(file)

    df = pd.DataFrame(event_data)

    del event_data
    gc.collect()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['jet_btag'])

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    train_dataset = JetParticleDataset(train_df, device=device, eval=False,scaler=True)
    test_dataset = JetParticleDataset(test_df, device=device, eval=True,scaler=True)

    print("Saving datasets...")
    torch.save(train_dataset, 'train_dataset_merged.pt')
    torch.save(test_dataset, 'test_dataset_merged.pt')

if __name__ == '__main__':
    main()
