import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from create_hitmatrix_and_labels import read_root_file

import uproot
print("Preping model")
root_file = "rootfiles/DY_Target_500k_080524/merged_trackQA_v2.root"
hitmatrix, Truth_valves, good_events= read_root_file(root_file)


#X is features and Y is labels. Train is for fitting and test is validation/ evaluation
X_train, X_test, y_train, y_test = train_test_split(hitmatrix, Truth_valves, test_size=0.25, random_state=42, shuffle=True)

checkpoint_filepath = 'checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = False,
    monitor = 'loss',
    mode = 'min',
    save_best_only = True)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=10,mode='min')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(49,201)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(4925,activation='relu'))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(392,activation='sigmoid'))
model.add(tf.keras.layers.Dense(196))

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=0.004)
loss_function = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size',name='mean_squared_error')
meteric_function = tf.keras.metrics.MeanSquaredError(name='mean_squared_error', dtype=None)

model.compile(loss=loss_function, optimizer=opt, metrics=[meteric_function])
hist = model.fit(X_train, y_train, epochs=5000, batch_size=16,validation_data=(X_test,y_test),callbacks=[callback])

model.save('hit_array_model')

Model_loss = hist.history['loss']
Val_loss = hist.history['val_loss']

plt.plot(Val_loss,color='k',label = 'val_loss')
plt.plot(Model_loss,color = 'r', label = "Model_loss")
#plt.ylim(0,100)
plt.xlabel("epochs")
plt.ylabel("Loss")
plt.title("valadation loss vs model loss")
plt.legend()
plt.savefig('Hit_array_loss_plot.png')

hit_arrays = model.predict(X_test)
number_of_events = len(X_test)

hit_arrays = np.reshape(hit_arrays,(number_of_events,49,4))
test = np.reshape(y_test,(number_of_events,49,4))
Model_Track_1_elemID = hit_arrays[:,:,0]
Test_Track_1_elemID = test[:,:,2]
Model_Track_2_elemID = hit_arrays[:,:,2]
Test_Track_2_elemID = test[:,:,2]

plt.clf()
plt.plot(Model_Track_1_elemID[0],color = 'r')
plt.plot(Test_Track_1_elemID[0])

plt.savefig('Error_plot.png')
