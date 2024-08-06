import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from create_hitmatrix_and_labels import read_root_file

root_file = "rootfiles/DY_Target_500k_080524/merged_trackQA_v2.root"
hitmatrix, Truth_valves = read_root_file(root_file)

#X is features and Y is labels. Train is for fitting and test is validation/ evaluation
X_train, X_test, y_train, y_test = train_test_split(hitmatrix, Truth_valves, test_size=0.25, random_state=42, shuffle=True)


model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(49,201)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(196))

model.summary()

opt = tf.keras.optimizers.Adam(learning_rate=1e-7)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
hist = model.fit(X_train, y_train, verbose=False, epochs=500)
plt.plot(hist.history['acc'])
plt.show()
