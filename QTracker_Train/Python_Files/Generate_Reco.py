import os
import sys
import numpy as np
import uproot
from numba import njit, prange
import random
import tensorflow as tf
import gc
from Common_Functions import *

if len(sys.argv) != 2:
        print("Usage: python script.py <Vertex Distribution>")
        print("Currently supports All, Z, Target, and Dump")
        exit(1)

vertex = sys.argv[1]
root_file_train = f"Root_Files/{vertex}_Train_QA_v2.root"
root_file_val = f"Root_Files/{vertex}_Val_QA_v2.root"

pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file(root_file_train)
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file(root_file_val)

pos_events=clean(pos_events).astype(int)
neg_events=clean(neg_events).astype(int)
pos_events_val=clean(pos_events_val).astype(int)
neg_events_val=clean(neg_events_val).astype(int)

@njit(parallel=True)
def track_injection(hits,drift,pos_e,neg_e,pos_d,neg_d,pos_k,neg_k):
    #Start generating the events
    kin=np.zeros((len(hits),9))
    for z in prange(len(hits)):
        j=random.randrange(len(pos_e))
        kin[z, :3] = pos_k[j, :3]
        kin[z, 3:9] = neg_k[j]
        for k in range(54):
            if(pos_e[j][k]>0):
                if(random.random()<0.94) and (k<30):
                    hits[z][k][int(pos_e[j][k]-1)]=1
                    drift[z][k][int(pos_e[j][k]-1)]=pos_d[j][k]
                if(k>29):
                    hits[z][k][int(pos_e[j][k]-1)]=1
            if(neg_e[j][k]>0):
                if(random.random()<0.94) and (k<30):
                    hits[z][k][int(neg_e[j][k]-1)]=1
                    drift[z][k][int(neg_e[j][k]-1)]=neg_d[j][k]
                if(k>29):
                    hits[z][k][int(neg_e[j][k]-1)]=1

    return hits,drift,kin

def generate_e906(n_events, tvt):
    #Create the realistic background for events
    hits, drift = build_background(n_events)
    #Place the full tracks that are reconstructable
    if(tvt=="Train"):
        hits,drift,kinematics=track_injection(hits,drift,pos_events,neg_events,pos_drift,neg_drift,pos_kinematics,neg_kinematics)    
    if(tvt=="Val"):
        hits,drift,kinematics=track_injection(hits,drift,pos_events_val,neg_events_val,pos_drift_val,neg_drift_val,pos_kinematics_val,neg_kinematics_val)    
    return hits.astype(bool), drift, kinematics

# Detect the number of GPUs available
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
print(f"Number of GPUs available: {num_gpus}")

# Set up strategy for distributed training
if num_gpus > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()
    
EF_batch = 256 * num_gpus
TF_batch = 64 * num_gpus
DN_batch = 8192 * num_gpus

def run_qtracker(hits, drift, kinematics):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/event_filter')
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        event_classification_probabilies = probability_model.predict(hits,batch_size=EF_batch, verbose=0)

    mask = event_classification_probabilies[:,1]>=0.75
    hits = hits[mask]
    drift = drift[mask]
    kinematics=kinematics[mask]
    event_classification_probabilies = event_classification_probabilies[mask]
    
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Track_Finder_Pos')
        pos_predictions = model.predict(hits, verbose=0, batch_size = TF_batch)
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Track_Finder_Neg')
        neg_predictions =model.predict(hits, verbose=0, batch_size = TF_batch)
    predictions = (np.round(np.column_stack((pos_predictions,neg_predictions))*max_ele)).astype(int)

    muon_tracks=evaluate_finder(hits,drift,predictions)

    tf.keras.backend.clear_session()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Vertexing_Pos')
        pos_pred = model.predict(muon_tracks[:,:34,:2], verbose=0, batch_size = DN_batch)
    tf.keras.backend.clear_session()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Vertexing_Neg')
        neg_pred = model.predict(muon_tracks[:,34:,:2], verbose=0, batch_size = DN_batch)

    muon_track_quality = calc_mismatches(muon_tracks)
    mask = ((muon_track_quality[0::4] < 2) & (muon_track_quality[1::4] < 2) & (muon_track_quality[2::4] < 3) & (muon_track_quality[3::4] < 3)).all(axis=0)

    # Apply the final filter to event_classification_probabilities
    hits = hits[mask]
    drift = drift[mask]
    muon_tracks = muon_tracks[mask]
    pos_pred = pos_pred[mask]
    neg_pred = neg_pred[mask]
    kinematics = kinematics[mask]
    muon_track_quality = muon_track_quality.T[mask]
    event_classification_probabilies = event_classification_probabilies[mask]

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Track_Finder_All')
        predictions = (np.round(model.predict(hits,verbose=0, batch_size = TF_batch)*max_ele)).astype(int)
    all_vtx_track = evaluate_finder(hits,drift,predictions)[:,:,:2]

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model=tf.keras.models.load_model('Networks/Reconstruction_All')
        reco_kinematics = model.predict(all_vtx_track,batch_size=DN_batch,verbose=0)

    vertex_input=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),all_vtx_track),axis=1)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model=tf.keras.models.load_model('Networks/Vertexing_All')
        reco_vertex = model.predict(vertex_input,batch_size=DN_batch,verbose=0)

    all_vtx_reco=np.concatenate((reco_kinematics,reco_vertex),axis=1)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Track_Finder_Z')
        predictions = (np.round(model.predict(hits,verbose=0, batch_size = TF_batch)*max_ele)).astype(int)
    z_vtx_track = evaluate_finder(hits,drift,predictions)[:,:,:2]

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model=tf.keras.models.load_model('Networks/Reconstruction_Z')
        reco_kinematics = model.predict(z_vtx_track,batch_size=DN_batch,verbose=0)

    vertex_input=np.concatenate((reco_kinematics.reshape((len(reco_kinematics),3,2)),z_vtx_track),axis=1)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model=tf.keras.models.load_model('Networks/Vertexing_Z')
        reco_vertex = model.predict(vertex_input,batch_size=DN_batch,verbose=0)

    z_vtx_reco=np.concatenate((reco_kinematics,reco_vertex),axis=1)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Track_Finder_Target')
        predictions = (np.round(model.predict(hits,verbose=0, batch_size = TF_batch)*max_ele)).astype(int)
    target_track = evaluate_finder(hits,drift,predictions)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model=tf.keras.models.load_model('Networks/Reconstruction_Target')
        target_vtx_reco = model.predict(target_track[:,:,:2],batch_size=DN_batch,verbose=0)

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model = tf.keras.models.load_model('Networks/Track_Finder_Dump')
        predictions = (np.round(model.predict(hits,verbose=0, batch_size = TF_batch)*max_ele)).astype(int)
    dump_track = evaluate_finder(hits,drift,predictions)[:,:,:2]

    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    with strategy.scope():
        model=tf.keras.models.load_model('Networks/Reconstruction_Dump')
        dump_vtx_reco = model.predict(dump_track,batch_size=DN_batch,verbose=0)
    
    dimuon_track_quality = calc_mismatches(target_track)

    mask = ((dimuon_track_quality[0::4] < 2) & (dimuon_track_quality[1::4] < 2) & (dimuon_track_quality[2::4] < 3) & (dimuon_track_quality[3::4] < 3)).all(axis=0)
    
    predictions = np.column_stack((event_classification_probabilies[:,1], pos_pred, neg_pred, all_vtx_reco, z_vtx_reco, target_vtx_reco, dump_vtx_reco, muon_track_quality, dimuon_track_quality.T))            
    tracks = np.column_stack((muon_tracks[:,:,:2], all_vtx_track[:,:,:2], z_vtx_track[:,:,:2], target_track[:,:,:2], dump_track[:,:,:2]))
    
    predictions = predictions[mask]
    tracks = tracks[mask]

    target_dump_input = np.column_stack((predictions,tracks.reshape((len(tracks),(68*2*5)))))

    return predictions, tracks

# Initialize lists to store the data
dimuon_probability=[]
all_predictions = []
tracks = []
truth = []
total_entries = 0
#Generate training data
while(total_entries<10000000):
    try:
        hits, drift, kinematics = generate_e906(500000,"Train")
        
        all_predictions, tracks = run_qtracker(hits, drift, kinematics)
        
        np.save(f'Training_Data/{vertex}_Tracks_Train.npy',np.concatenate(tracks, axis=0))
        np.save(f'Training_Data/{vertex}_Reco_Train.npy',np.concatenate(all_predictions, axis=0))

        total_entries += len(hits)
        print(total_entries)
        del hits, drift, all_predictions, tracks
    except Exception as e:
        pass        
        
# Initialize lists to store the data
dimuon_probability=[]
all_predictions = []
tracks = []
truth = []
total_entries = 0


#Generate validation data
while(total_entries<1000000):
    try:
        hits, drift, kinematics = generate_e906(500000,"Val")
        
        all_predictions, tracks = run_qtracker(hits, drift, kinematics)
            
        np.save(f'Training_Data/{vertex}_Tracks_Val.npy',np.concatenate(tracks, axis=0))
        np.save(f'Training_Data/{vertex}_Reco_Val.npy',np.concatenate(all_predictions, axis=0))
        
        total_entries += len(hits)
        print(total_entries)
        del hits, drift, all_predictions, tracks
    except Exception as e:
        pass
