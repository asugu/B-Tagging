import uproot as uproot
import numpy as np
import awkward as ak
import gc
import time
from datetime import timedelta
import warnings

import pickle

warnings.filterwarnings("ignore", category=DeprecationWarning)

def pad_array(array, target_length, default_value=0, reverse=False):

    if len(array) >= target_length:
        array = array[:target_length] 

    else:    
        array = np.pad(array, (0, target_length - len(array)), constant_values=default_value)

    if reverse:
        array = array[::-1]

    return array

def sorter_(unsorted_array, sorted_indices):
    
    sorted_array = ak.Array([[unsorted_array[i][sorted_indices[i][j]] for j in range(len(unsorted_array[i]))]for i in range(len(unsorted_array)) ])

    return sorted_array

def root_processor(tree, key=1):

    branch_iEvent = tree['iEvent']
    branch_hadron_flav = tree["hadron_flavour"]
    branch_parton_flav = tree["parton_flavour"]
    branch_gen_match_flav = tree["matched_gen_flavour"]

    branch_jet_mass = tree["jet_mass"]
    branch_jet_pt = tree["jet_pt"]
    branch_jet_eta = tree["jet_eta"]
    branch_jet_phi = tree["jet_phi"]

    branch_track_E = tree["track_E"]
    branch_track_pt = tree["track_pt"]
    branch_track_pid = tree["track_pdgId"]
    branch_track_charge = tree["track_charge"]
    branch_track_d0 = tree["track_dxy"]
    branch_track_dz = tree["track_dz"]
    branch_track_d0_sig = tree["track_dxy_sig"]
    branch_track_dz_sig = tree["track_dz_sig"]
    branch_track_deta = tree["track_deta"]
    branch_track_dphi = tree["track_dphi"]

    branch_sv_pt = tree["sv_pt"]
    branch_sv_charge = tree["sv_charge"]
    branch_sv_ntracks = tree["sv_ntracks"]
    branch_sv_mass = tree["sv_mass"]
    branch_sv_chi2 = tree["sv_chi2"]
    branch_sv_ndof = tree["sv_ndof"]
    branch_sv_dxy = tree["sv_dxy"]
    branch_sv_dlen = tree["sv_dlen"]
    branch_sv_dxy_sig = tree["sv_dxy_sig"]
    branch_sv_dlen_sig = tree["sv_dlen_sig"]

    iEvent = ak.Array(branch_iEvent.array())
    hadron_flav = ak.Array(branch_hadron_flav.array())
    parton_flav = ak.Array(branch_parton_flav.array())
    gen_match_flav = ak.Array(branch_gen_match_flav.array())

    jet_mass = ak.Array(branch_jet_mass.array())
    jet_pt = ak.Array(branch_jet_pt.array())
    jet_eta = ak.Array(branch_jet_eta.array())
    jet_phi = ak.Array(branch_jet_phi.array())

    unsorted_track_E = ak.Array(branch_track_E.array())
    unsorted_track_pt = ak.Array(branch_track_pt.array())
    unsorted_track_pid = ak.Array(branch_track_pid.array())
    unsorted_track_charge = ak.Array(branch_track_charge.array())
    unsorted_track_d0 = ak.Array(branch_track_d0.array())
    unsorted_track_dz = ak.Array(branch_track_dz.array())
    unsorted_track_d0_sig = ak.Array(branch_track_d0_sig.array())
    unsorted_track_dz_sig = ak.Array(branch_track_dz_sig.array())
    unsorted_track_deta = ak.Array(branch_track_deta.array())
    unsorted_track_dphi = ak.Array(branch_track_dphi.array())

    unsorted_sv_pt = ak.Array(branch_sv_pt.array())
    unsorted_sv_charge = ak.Array(branch_sv_charge.array())
    unsorted_sv_ntracks = ak.Array(branch_sv_ntracks.array())
    unsorted_sv_mass = ak.Array(branch_sv_mass.array())
    unsorted_sv_chi2 = ak.Array(branch_sv_chi2.array())
    unsorted_sv_ndof = ak.Array(branch_sv_ndof.array())
    unsorted_sv_dxy = ak.Array(branch_sv_dxy.array())
    unsorted_sv_dlen = ak.Array(branch_sv_dlen.array())
    unsorted_sv_dxy_sig = ak.Array(branch_sv_dxy_sig.array())
    unsorted_sv_dlen_sig = ak.Array(branch_sv_dlen_sig.array())
                            
    jet_btag_true = [1 if value == 5 else 0 for value in hadron_flav]

    print("total jets: ", len(hadron_flav))

    no_flav = 0
    for i in range(len(parton_flav)):
        if parton_flav[i] == 5:
            no_flav += 1
    print("number of b in parton_flav: ", no_flav)

    no_flav = 0
    for i in range(len(gen_match_flav)):
        if gen_match_flav[i] == 5:
            no_flav += 1
    print("number of b in gen_match_flav: ", no_flav)

    no_flav = 0
    for i in range(len(hadron_flav)):
        if hadron_flav[i] == 5:
            no_flav += 1 
    print("number of b in hadron_flav: ", no_flav)

    no_ctag = 0
    for i in range(len(hadron_flav)):
        if hadron_flav[i] == 4:
            no_ctag += 1
    print("number of c_tags: ", no_ctag)


    ##############################################
    ###########    Track Processing    ###########

    pt_sorted_indices = ak.argsort(unsorted_track_pt, axis=1, ascending=False,stable=True)

    max_track_length = max(len(track) for track in unsorted_track_E)

    print("max track length is : ", max_track_length)

    track_E = sorter_(unsorted_track_E,sorted_indices=pt_sorted_indices)
    #track_E = sorter_(unsorted_track_E,sorted_indices=pt_sorted_indices)






    track_pt = sorter_(unsorted_track_pt,sorted_indices=pt_sorted_indices)
    track_pid = sorter_(unsorted_track_pid,sorted_indices=pt_sorted_indices)
    track_charge = sorter_(unsorted_track_charge,sorted_indices=pt_sorted_indices)
    track_d0 = sorter_(unsorted_track_d0,sorted_indices=pt_sorted_indices)
    track_dz = sorter_(unsorted_track_dz,sorted_indices=pt_sorted_indices)
    track_d0_sig = sorter_(unsorted_track_d0_sig,sorted_indices=pt_sorted_indices)
    track_dz_sig = sorter_(unsorted_track_dz_sig,sorted_indices=pt_sorted_indices)
    track_deta = sorter_(unsorted_track_deta,sorted_indices=pt_sorted_indices)
    track_dphi = sorter_(unsorted_track_dphi,sorted_indices=pt_sorted_indices)

    del unsorted_track_E, unsorted_track_pt, unsorted_track_deta, unsorted_track_dphi, unsorted_track_pid, unsorted_track_charge, unsorted_track_d0, unsorted_track_dz, unsorted_track_d0_sig, unsorted_track_dz_sig
    gc.collect()


    track_count = []
    for i in range(len(track_E)):
        track_count.append(len(track_E[i]))

    non_empty_track_indices = [i for i, track in enumerate(track_E) if len(track) > 0]
    print("non_empty_track_indices : ",len(non_empty_track_indices))

    max_track_length = 16

    track_E = [pad_array(track, max_track_length) for track in track_E]
    track_pt = [pad_array(track, max_track_length) for track in track_pt]
    track_pid = [pad_array(track, max_track_length) for track in track_pid]
    track_charge = [pad_array(track, max_track_length) for track in track_charge]
    track_d0 = [pad_array(track, max_track_length) for track in track_d0]
    track_dz = [pad_array(track, max_track_length) for track in track_dz]
    track_d0_sig = [pad_array(track, max_track_length) for track in track_d0_sig]
    track_dz_sig = [pad_array(track, max_track_length) for track in track_dz_sig]
    track_deta = [pad_array(track, max_track_length) for track in track_deta]
    track_dphi = [pad_array(track, max_track_length) for track in track_dphi]



    #########################################################
    ###########    Secondary Vertex Processing    ###########

    max_sv_length = max(len(sv) for sv in unsorted_sv_pt)

    print("max sv length is : ", max_sv_length)

    max_sv_length = 5


    pt_sorted_indices = ak.argsort(unsorted_sv_pt, axis=1, ascending=False,stable=True)

    sv_pt = sorter_(unsorted_sv_pt,sorted_indices=pt_sorted_indices)
    sv_charge = sorter_(unsorted_sv_charge,sorted_indices=pt_sorted_indices)
    sv_ntracks = sorter_(unsorted_sv_ntracks,sorted_indices=pt_sorted_indices)
    sv_mass = sorter_(unsorted_sv_mass,sorted_indices=pt_sorted_indices)
    sv_chi2 = sorter_(unsorted_sv_chi2,sorted_indices=pt_sorted_indices)
    sv_ndof = sorter_(unsorted_sv_ndof,sorted_indices=pt_sorted_indices)
    sv_dxy = sorter_(unsorted_sv_dxy,sorted_indices=pt_sorted_indices)
    sv_dlen = sorter_(unsorted_sv_dlen,sorted_indices=pt_sorted_indices)
    sv_dxy_sig = sorter_(unsorted_sv_dxy_sig,sorted_indices=pt_sorted_indices)
    sv_dlen_sig = sorter_(unsorted_sv_dlen_sig,sorted_indices=pt_sorted_indices)

    del unsorted_sv_pt, unsorted_sv_charge, unsorted_sv_ntracks, unsorted_sv_mass, unsorted_sv_chi2, unsorted_sv_ndof, unsorted_sv_dxy, unsorted_sv_dlen, unsorted_sv_dxy_sig, unsorted_sv_dlen_sig
    gc.collect()

    sv_count = []
    for i in range(len(sv_pt)):
        sv_count.append(len(sv_pt[i]))

    non_empty_sv_indices = [i for i, vertex in enumerate(sv_pt) if len(vertex) > 0]
    print("non_empty_sv_indices : ",len(non_empty_sv_indices))

    sv_pt = [pad_array(vertex, max_sv_length) for vertex in sv_pt]
    sv_charge = [pad_array(vertex, max_sv_length) for vertex in sv_charge]
    sv_ntracks = [pad_array(vertex, max_sv_length) for vertex in sv_ntracks]
    sv_mass = [pad_array(vertex, max_sv_length) for vertex in sv_mass]
    sv_chi2 = [pad_array(vertex, max_sv_length) for vertex in sv_chi2]
    sv_ndof = [pad_array(vertex, max_sv_length) for vertex in sv_ndof]
    sv_dxy = [pad_array(vertex, max_sv_length) for vertex in sv_dxy]
    sv_dlen = [pad_array(vertex, max_sv_length) for vertex in sv_dlen]
    sv_dxy_sig = [pad_array(vertex, max_sv_length) for vertex in sv_dxy_sig]
    sv_dlen_sig = [pad_array(vertex, max_sv_length) for vertex in sv_dlen_sig]

    ####################################################
    ###########    Output File Processing    ###########

   # non_empty_both_indices = [i for i in non_empty_sv_indices if i in non_empty_track_indices]


    non_empty_both_indices = [
        i for i in non_empty_sv_indices
        if i in non_empty_track_indices and abs(jet_eta[i]) < 2.5
    ]

    # non_empty_both_indices = [
    #     i for i in non_empty_track_indices if abs(jet_eta[i]) < 2.5
    # ]

    print("Number of jets where tracks are non-empty with eta < 2.5:", len(non_empty_both_indices))

    #print("Number of jets where both SV and tracks are non-empty:", len(non_empty_both_indices))

    no_flav = 0
    for i in non_empty_both_indices:
        if parton_flav[i] == 5:
            no_flav += 1
    print("number of b in parton_flav: ", no_flav)

    no_flav = 0
    for i in non_empty_both_indices:
        if gen_match_flav[i] == 5:
            no_flav += 1
    print("number of b in gen_match_flav: ", no_flav)

    no_flav = 0
    for i in non_empty_both_indices:
        if hadron_flav[i] == 5:
            no_flav += 1 
    print("number of b in hadron_flav: ", no_flav)

    no_ctag = 0
    for i in non_empty_both_indices:
        if hadron_flav[i] == 4:
            no_ctag += 1
    print("number of c_tags: ", no_ctag)

    event_data = []
    for i in non_empty_both_indices:
        event_dict = {

            'iEvent': iEvent[i],
            'hadron_flav': hadron_flav[i],
            'parton_flav': parton_flav[i],
            'gen_match_flav': gen_match_flav[i],
            'jet_btag' : jet_btag_true[i], # binary b tag

            'jet_mass' : jet_mass[i],
            'jet_pt' : jet_pt[i],
            'jet_eta' : jet_eta[i],
            'jet_phi' : jet_phi[i],

            'jet_track_count': track_count[i],
            'jet_sv_count': sv_count[i],

            'track_E': np.array(track_E[i], dtype=np.float32),
            'track_pt': np.array(track_pt[i], dtype=np.float32),
            'track_pid': np.array(track_pid[i], dtype=np.float32),
            'track_charge': np.array(track_charge[i], dtype=np.float32),
            'track_d0': np.array(track_d0[i], dtype=np.float32),
            'track_dz': np.array(track_dz[i], dtype=np.float32),
            'track_d0_sig': np.array(track_d0_sig[i], dtype=np.float32),
            'track_dz_sig': np.array(track_dz_sig[i], dtype=np.float32),
            'track_deta': np.array(track_deta[i], dtype=np.float32),
            'track_dphi': np.array(track_dphi[i], dtype=np.float32),

            'sv_pt' : np.array(sv_pt[i], dtype=np.float32),
            #'sv_charge' : np.array(sv_charge[i], dtype=np.float32),
            'sv_ntracks' : np.array(sv_ntracks[i], dtype=np.float32),
            'sv_mass' : np.array(sv_mass[i], dtype=np.float32),
            'sv_chi2' : np.array(sv_chi2[i], dtype=np.float32),
            'sv_ndof' : np.array(sv_ndof[i], dtype=np.float32),
            'sv_dxy' : np.array(sv_dxy[i], dtype=np.float32),
            'sv_dlen' : np.array(sv_dlen[i], dtype=np.float32),
            'sv_dxy_sig' : np.array(sv_dxy_sig[i], dtype=np.float32),
            'sv_dlen_sig' : np.array(sv_dlen_sig[i], dtype=np.float32),

        }
        event_data.append(event_dict)

    #file_path = 'data/event_data_ttsemi_pad25_2.pkl'
    file_path = f'data/event_data_sv_pad({max_sv_length}_{max_track_length})_{key}.pkl'

    with open(file_path, 'wb') as file:
        pickle.dump(event_data, file)
    
    del event_dict
    gc.collect()

if __name__ == "__main__":
    
    root_file_path = "data/output_sv.root"
    #root_file_path = "data/output_bjet_analysis_short.root"   # For testing

    root_file = uproot.open(root_file_path)

    tree_names = root_file.keys()
    print(tree_names)
    tree_keys = [name.split(';')[1] for name in tree_names if name.startswith('BJetAnalyzer/JetTree;')]

    for key in tree_keys:
        start_time = time.time()
        print(f"Processing tree number {key}...")

        tree = root_file[f'BJetAnalyzer/JetTree;{key}']
        root_processor(tree, key)

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Total processing time of tree {key}: {total_time:.2f} seconds")
        total_duration = timedelta(seconds=total_time)
        print(f"Total processing time of tree {key}: {total_duration}")