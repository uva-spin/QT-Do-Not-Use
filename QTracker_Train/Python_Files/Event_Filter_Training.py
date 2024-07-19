import os
import numpy as np
import uproot
from numba import njit, prange
import random
import tensorflow as tf
import gc
from Common_Functions import *

@njit(parallel=True)
def track_injection(hits,pos_e,neg_e):
    # Inject tracks into the hit matrices
    category=np.zeros((len(hits)))
    for z in prange(len(hits)):
        m = random.randrange(0,2)
        j=random.randrange(len(pos_e))
        for k in range(54):
            if(pos_e[j][k]>0):
                if(random.random()<m*0.94) or ((k>29)&(k<45)):
                    hits[z][k][int(pos_e[j][k]-1)]=1
            if(neg_e[j][k]>0):
                if(random.random()<m*0.94) or ((k>29)&(k<45)):
                    hits[z][k][int(neg_e[j][k]-1)]=1
        category[z]=m        

    return hits,category

def generate_hit_matrices(n_events, tvt):
    #Create the realistic background for events
    hits, _ = build_background(n_events)
    #Inject the reconstructable tracks
    if(tvt=="Train"):
        hits,category=track_injection(hits,pos_events,neg_events)    
    if(tvt=="Val"):
        hits,category=track_injection(hits,pos_events_val,neg_events_val)    
    return hits.astype(bool), category.astype(int)

# Read training and validation data from ROOT files
pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics = read_root_file('Root_Files/Target_Train_QA_v2.root')
pos_events_val, pos_drift_val, pos_kinematics_val, neg_events_val, neg_drift_val, neg_kinematics_val = read_root_file('Root_Files/Target_Val_QA_v2.root')

del pos_drift, neg_drift, pos_kinematics, neg_kinematics
del pos_drift_val, neg_drift_val, pos_kinematics_val, neg_kinematics_val

# Clean event data by setting values > 1000 to 0.
pos_events=clean(pos_events).astype(int)
neg_events=clean(neg_events).astype(int)
pos_events_val=clean(pos_events_val).astype(int)
neg_events_val=clean(neg_events_val).astype(int)

# Set learning rate and callback for early stopping
learning_rate_filter = 1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
n_train = 0

# Detect the number of GPUs available
gpus = tf.config.experimental.list_physical_devices('GPU')
num_gpus = len(gpus)
print(f"Number of GPUs available: {num_gpus}")

# Set up strategy for distributed training
if num_gpus > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()
    
# Adjust batch size for the number of GPUs
batch_size_adjusted = 256 * num_gpus

print("Before while loop:", n_train)
while n_train < 1e7:
    # Generate training and validation data
    trainin, trainsignals = generate_hit_matrices(1000000, "Train")
    n_train += len(trainin)
    print("Generated Training Data")
    valin, valsignals = generate_hit_matrices(100000, "Val")
    print("Generated Validation Data")
    
    # Clear session and reset TensorFlow graph
    tf.keras.backend.clear_session()
    gc.collect()
    with strategy.scope():
        # Load and compile the model
        model = tf.keras.models.load_model('Networks/event_filter')
        optimizer = tf.keras.optimizers.Adam(learning_rate_filter)
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        # Evaluate the model before training
        val_loss_before = model.evaluate(valin, valsignals, batch_size=batch_size_adjusted, verbose=2)[0]

        # Train the model
        history = model.fit(trainin, trainsignals,
                            epochs=1000, batch_size=batch_size_adjusted, verbose=2, 
                            validation_data=(valin, valsignals), callbacks=[callback])

        # Check if the validation loss improved
        if min(history.history['val_loss']) < val_loss_before:
            model.save('Networks/event_filter')
            learning_rate_filter *= 2
        learning_rate_filter /= 2

    del trainsignals, trainin, valin, valsignals, model
    gc.collect()
    print(n_train)


