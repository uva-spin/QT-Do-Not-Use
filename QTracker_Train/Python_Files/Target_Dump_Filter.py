import numpy as np
import tensorflow as tf
import os

# Load data
target_reco_train = np.load('Training_Data/Target_Reco_Train.npy')
target_tracks_train = np.load('Training_Data/Target_Tracks_Train.npy')
filt = np.max(abs(target_reco_train),axis=1)<1000
target_reco_train = target_reco_train[filt]
target_tracks_train = target_tracks_train[filt]
target_train_data = np.column_stack((target_reco_train,target_tracks_train.reshape((len(target_tracks_train),(68*2*5)))))
del target_reco_train, target_tracks_train
target_train_labels = np.ones(len(target_train_data))

target_reco_val = np.load('Training_Data/Target_Reco_Val.npy')
target_tracks_val = np.load('Training_Data/Target_Tracks_Val.npy')
filt = np.max(abs(target_reco_val),axis=1)<1000
target_reco_val = target_reco_val[filt]
target_tracks_val = target_tracks_val[filt]
target_val_data = np.column_stack((target_reco_val,target_tracks_val.reshape((len(target_tracks_val),(68*2*5)))))
del target_reco_val, target_tracks_val
target_val_labels = np.ones(len(target_val_data))


# Load data
dump_reco_train = np.load('Training_Data/Dump_Reco_Train.npy')
dump_tracks_train = np.load('Training_Data/Dump_Tracks_Train.npy')
filt = np.max(abs(dump_reco_train),axis=1)<1000
dump_reco_train = dump_reco_train[filt]
dump_tracks_train = dump_tracks_train[filt]
dump_train_data = np.column_stack((dump_reco_train,dump_tracks_train.reshape((len(dump_tracks_train),(68*2*5)))))
del dump_reco_train, dump_tracks_train
dump_train_labels = np.zeros(len(dump_train_data))

dump_reco_val = np.load('Training_Data/Dump_Reco_Val.npy')
dump_tracks_val = np.load('Training_Data/Dump_Tracks_Val.npy')
filt = np.max(abs(dump_reco_val),axis=1)<1000
dump_reco_val = dump_reco_val[filt]
dump_tracks_val = dump_tracks_val[filt]
dump_val_data = np.column_stack((dump_reco_val,dump_tracks_val.reshape((len(dump_tracks_val),(68*2*5)))))
del dump_reco_val, dump_tracks_val
dump_val_labels = np.zeros(len(dump_val_data))

# Combine data and labels
X_train = np.concatenate((target_train_data, dump_train_data), axis=0)
del target_train_data, dump_train_data
y_train = np.concatenate((target_train_labels, dump_train_labels), axis=0)
del target_train_labels, dump_train_labels

indices = np.arange(len(X_train))
np.random.shuffle(indices)

X_train = X_train[indices]
y_train = y_train[indices]

# Combine data and labels
X_val = np.concatenate((target_val_data, dump_val_data), axis=0)
del target_val_data, dump_val_data
y_val = np.concatenate((target_val_labels, dump_val_labels), axis=0)
del target_val_labels, dump_val_labels

indices = np.arange(len(X_val))
np.random.shuffle(indices)

X_val = X_val[indices]
y_val = y_val[indices]


model = tf.keras.models.load_model('Networks/target_dump_filter')

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

# Compile the model with the updated optimizer
model.compile(optimizer=optimizer,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
# Set up early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, epochs=10000, batch_size=512, validation_data=(X_val, y_val),
                    verbose=2, callbacks=[early_stopping])

# Save the model
model.save('Networks/target_dump_filter')
