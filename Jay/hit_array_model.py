import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set paths to data. Makes a list of npz files for trainig.
data_folder = "/Users/jay/Documents/Research/machine_learning/Hit_Data"
npz_files = [os.path.join(data_folder, file) for file in os.listdir(data_folder) if file.endswith('.npz')]

# Function to load and process data from npz files in batches
def load_data_in_batches(npz_files, batch_size=64):
    for npz_file in npz_files:
        data = np.load(npz_file)
        hit_matrix = data['hit_matrix']  # Shape (num_samples, 62, 201)
        Truth_elementID_mup = data['Truth_elementID_mup']  # Labels for output_1
        Truth_elementID_mum = data['Truth_elementID_mum']  # Labels for output_2
        
        #Array of detector indexes we are not using.
        indices_array = np.array([6,7,8,9,10,11,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])
        # Delete the columns from Labels to simplify Array
        Truth_elementID_mum = np.delete(Truth_elementID_mum, indices_array, axis=1)
        Truth_elementID_mup = np.delete(Truth_elementID_mup, indices_array, axis=1)
        hit_matrix = np.delete(hit_matrix, indices_array, axis=1) # Shape (num_samples, 40, 201)

        # Split into training and testing sets
        X_train, X_test, y_train_mup, y_test_mup = train_test_split(
            hit_matrix, Truth_elementID_mup, test_size=0.25, random_state=42, shuffle=True
        )
        _, _, y_train_mum, y_test_mum = train_test_split(
            hit_matrix, Truth_elementID_mum, test_size=0.25, random_state=42, shuffle=True
        )

        # Create labels as a dictionary for multi-output training
        y_train = {'output_1': y_train_mup, 'output_2': y_train_mum}
        y_test = {'output_1': y_test_mup, 'output_2': y_test_mum}

        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))

        # Shuffle, batch, and prefetch for efficiency
        train_dataset = train_dataset.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

        yield train_dataset, test_dataset



# Create a simple DNN model
def build_model():

    # Define the input layer for hitmatrix input
    input_layer = Input(shape=(40, 201))  # Example input shape

    # Flatten the 2D input into a 1D vector to use in a DNN
    flattened_input = layers.Flatten()(input_layer)

    # Shared base of the DNN
    x = layers.Dense(2048, activation='relu')(flattened_input)
    x = layers.Dropout(0.5)(x)  # Dropout layer with 50% dropout rate
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)


    # First head for Mup
    head_1 = layers.Dense(128, activation='relu')(x)
    #head_1 = layers.Dropout(0.3)(head_1)
    output_1 = layers.Dense(40, activation='linear', name='output_1')(head_1)  

    # Second head for Mum
    head_2 = layers.Dense(128, activation='relu')(x)
    #head_2 = layers.Dropout(0.3)(head_2)
    output_2 = layers.Dense(40, activation='linear', name='output_2')(head_2)  
    
    model = Model(inputs=input_layer, outputs=[output_1, output_2])

    # Compile the model with separate loss functions for each head
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=.001),
        loss={'output_1': 'mse', 'output_2': 'mse'},
        metrics={'output_1': tf.keras.metrics.RootMeanSquaredError(), 'output_2': tf.keras.metrics.RootMeanSquaredError()})
    return model


# Main training loop for multiple files
model = build_model()
model.summary()

history_all = {
    'loss_output_1': [], 
    'val_loss_output_1': [], 
    # 'accuracy_output_1': [], 
    # 'val_accuracy_output_1': [],
    'loss_output_2': [], 
    'val_loss_output_2': [], 
    # 'accuracy_output_2': [], 
    # 'val_accuracy_output_2': []
}
# Loop over npz files, training incrementally on each
for train_dataset, test_dataset in load_data_in_batches(npz_files, batch_size=32):
    # Fit model on the current batch of data
    history = model.fit(train_dataset, validation_data=test_dataset, epochs=5)

    # Append the loss and accuracy for both outputs to history_all
    history_all['loss_output_1'].extend(history.history['output_1_loss'])
    history_all['val_loss_output_1'].extend(history.history['val_output_1_loss'])
    # history_all['accuracy_output_1'].extend(history.history['output_1_accuracy'])
    # history_all['val_accuracy_output_1'].extend(history.history['val_output_1_accuracy'])

    history_all['loss_output_2'].extend(history.history['output_2_loss'])
    history_all['val_loss_output_2'].extend(history.history['val_output_2_loss'])
    # history_all['accuracy_output_2'].extend(history.history['output_2_accuracy'])
    # history_all['val_accuracy_output_2'].extend(history.history['val_output_2_accuracy'])


model.save("Version_8")

# Plotting the training and validation loss and accuracy for both outputs and save as PNG
plt.figure(figsize=(10, 12))

# Plot Loss for Mup
plt.subplot(2, 1, 1)  # Create a subplot for loss
plt.plot(history_all['loss_output_1'], label='Training Loss MuP')
plt.plot(history_all['val_loss_output_1'], label='Validation Loss MuP')
plt.title('Training and Validation Loss for Positive Muon over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 40)
plt.legend()
plt.grid(True)

# # Plot Accuracy for MuM
# plt.subplot(4, 1, 2)  # Create a subplot for accuracy
# plt.plot(history_all['accuracy_output_1'], label='Training Accuracy MuP')
# plt.plot(history_all['val_accuracy_output_1'], label='Validation Accuracy MuP')
# plt.title('Training and Validation Accuracy for Positive Muon over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.ylim(0, 1)  # Assuming accuracy is a percentage
# plt.legend()
# plt.grid(True)

# Plot Loss for MuM
plt.subplot(2, 1, 2)  # Create a subplot for loss
plt.plot(history_all['loss_output_2'], label='Training Loss MuM')
plt.plot(history_all['val_loss_output_2'], label='Validation Loss MuM')
plt.title('Training and Validation Loss for Negative Muon over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(0, 40)
plt.legend()
plt.grid(True)

# # Plot Accuracy for MuM
# plt.subplot(4, 1, 4)  # Create a subplot for accuracy
# plt.plot(history_all['accuracy_output_2'], label='Training Accuracy MuM')
# plt.plot(history_all['val_accuracy_output_2'], label='Validation Accuracy MuM')
# plt.title('Training and Validation Accuracy for Negative Muon over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.ylim(0, 1)  # Assuming accuracy is a percentage
# plt.legend()
# plt.grid(True)

# Save the plot as PNG file
plt.tight_layout()  # Adjust subplots to fit into figure area.
plt.savefig('accuracy_loss_plot_both_outputs.png')

# Close the plot to free up memory
plt.close()
