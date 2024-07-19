import numpy as np
import tensorflow as tf
import gc
import sys

if len(sys.argv) != 3:
    print("Usage: python script.py <Option> <Version>")
    print("Options are Vertex or Momentum")
    print("Version are Pos, Neg, All, Z, Target, or Dump")
    exit(1)

opt = sys.argv[1]
vers = sys.argv[2]
version = sys.argv[2]

if(vers == 'Pos') or (vers == 'Neg'):
    single_muon=True
else:single_muon=False

if opt == 'Vertex':
    model_name = f'Networks/Vertexing_{version}'
    mom_model_name = f'Networks/Reconstruction_{version}'
    
if opt == 'Momentum':
    model_name = f'Networks/Reconstruction_{version}'
    if single_muon: 
        print('Momentum reconstruction not implemented for single-muons.')

if single_muon: version='Muon'

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
batch_size_training = 1024 * num_gpus 

#Define the means and standard deviations for output normalization
kin_means = np.array([2,0,35,-2,0,35])
kin_stds = np.array([0.6,1.2,10,0.6,1.2,10])
vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

#Define the learning rate and callback
learning_rate=1e-6
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=100, restore_best_weights=True)

#Load the pre-generated training data
valin_reco = np.load(f"Training_Data/{version}_Val_In.npy")
valkinematics = np.load(f"Training_Data/{version}_Val_Out.npy")
filt = np.max(abs(valin_reco.reshape(len(valin_reco),(136))),axis=1)<1000
valin_reco = valin_reco[filt]
valkinematics = valkinematics[filt]

trainin_reco = np.load(f"Training_Data/{version}_Train_In.npy")
trainkinematics = np.load(f"Training_Data/{version}_Train_Out.npy")
filt = np.max(abs(trainin_reco.reshape(len(trainin_reco),(136))),axis=1)<1000
trainin_reco = trainin_reco[filt]
trainkinematics = trainkinematics[filt]

if opt == 'Vertex':
    if(vers=="Pos"):
        trainout = trainkinematics[:,0]
        valout = valkinematics[:,0]
    if(vers=="Neg"):
        trainout = trainkinematics[:,1]
        valout = valkinematics[:,1]
    if ~single_muon:
        trainout = trainkinematics[:,-3:]
        valout = valkinematics[:,-3:]
        trainout = (trainout-vertex_means)/vertex_stds
        valout = (valout-vertex_means)/vertex_stds
        with strategy.scope():
            model=tf.keras.models.load_model(mom_model_name)

            train_reco = model.predict(trainin_reco, verbose=0, batch_size = 8192*num_gpus)
            val_reco = model.predict(valin_reco, verbose=0, batch_size = 8192*num_gpus)

        trainin_reco=np.concatenate((train_reco.reshape((len(train_reco),3,2)),
                                     trainin_reco),axis=1)
        valin_reco=np.concatenate((val_reco.reshape((len(val_reco),3,2)),
                                   valin_reco),axis=1)


if opt == 'Momentum':
    trainout = np.column_stack((trainkinematics[:,:3],trainkinematics[:,-6:-3]))
    valout = np.column_stack((valkinematics[:,:3],valkinematics[:,-6:-3]))
    trainout = (trainout-kin_means)/kin_stds
    valout = (valout-kin_means)/kin_stds

tf.keras.backend.clear_session()
with strategy.scope():
    model=tf.keras.models.load_model(model_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer,
          loss=tf.keras.losses.mse,
          metrics=tf.keras.metrics.RootMeanSquaredError())
    history = model.fit(trainin_reco, trainout,
                epochs=10000, batch_size=batch_size_training, verbose=2, 
                validation_data=(valin_reco,valout),callbacks=[callback])
    model.save(model_name)
