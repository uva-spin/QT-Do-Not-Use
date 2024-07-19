import tensorflow as tf

#Event Filter

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(54, 201, 1)),
    tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), padding='valid', activation='relu'),
    tf.keras.layers.MaxPooling2D((1, 6), strides=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((1, 6), strides=(2, 2), padding='valid'),
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((1, 6), strides=(2, 2), padding='valid'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2)
])

# Save the model
model.save('Networks/event_filter')


#Track Finder Networks
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', input_shape=(54,201,1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(34, activation='linear')
    ])
#Save the individual muon track finders.
model.save('Networks/Track_Finder_Pos')
model.save('Networks/Track_Finder_Neg')

#Change the output layer to shape 68 to make it work for dimuon track finding.
model.pop()  # Remove the final layer
model.add(tf.keras.layers.Dense(68, activation='linear')) 

#Save the dimuon track finders.
model.save('Networks/Track_Finder_All')
model.save('Networks/Track_Finder_Z')
model.save('Networks/Track_Finder_Target')
model.save('Networks/Track_Finder_Dump')

# Define Kinematic Reconstruction Networks
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(68,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(6)])

#Save a copy each for all vertices, z vertices, and for target vertices.
model.save('Networks/Reconstruction_All')
model.save('Networks/Reconstruction_Z')
model.save('Networks/Reconstruction_Target')
model.save('Networks/Reconstruction_Dump')

#Define the single muon vertex finding networks.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(34,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(1)])

#Save the individual muon vertex finders.
model.save('Networks/Vertexing_Pos')
model.save('Networks/Vertexing_Neg')

#Define the dimuon vertex finding networks.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(71,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation='relu'),
    tf.keras.layers.Dense(256,activation='relu'),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(64,activation='relu'),
    tf.keras.layers.Dense(32,activation='relu'),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(3)])

#Save a copy each for all vertices and for Z vertices.
model.save('Networks/Vertexing_All')
model.save('Networks/Vertexing_Z')


# Define the target-dump filter network.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(757,)),
    tf.keras.layers.Dense(2048, activation='relu'),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2)
])

#Save the model for training.
model.save('Networks/target_dump_filter')