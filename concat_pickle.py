import pickle

pad = 64
rev = ''#'_reversed'

with open(f'data/event_data_sv_pad(5_16)_33.pkl', 'rb') as f1:
    data1 = pickle.load(f1)

with open(f'data/event_data_sv_pad(5_16)_32.pkl', 'rb') as f2:
    data2 = pickle.load(f2)


concatenated_data = data1 + data2

with open(f'data/event_data_sv_pad(5_16)_merged.pkl', 'wb') as f:
    pickle.dump(concatenated_data, f)
