import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns





# Load the model using Keras API
model = tf.keras.models.load_model("/Users/jay/Documents/Research/machine_learning/Model_V8/Version_8")

# Load data from the .npz file
with np.load('/Users/jay/Documents/Research/machine_learning/Eval_Data/Hit_Data/Hit_Info_DY_Target_5M_batch_1.npz') as data:
    hit_matrix = data['hit_matrix']  
    Truth_elementID_mup = data['Truth_elementID_mup']  
    Truth_elementID_mum = data['Truth_elementID_mum']  

    #Array of detector indexes we are not using.
    indices_array = np.array([6,7,8,9,10,11,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61])
    # Delete the columns from Labels to simplify Array
    Truth_elementID_mum = np.delete(Truth_elementID_mum, indices_array, axis=1)
    Truth_elementID_mup = np.delete(Truth_elementID_mup, indices_array, axis=1)
    hit_matrix = np.delete(hit_matrix, indices_array, axis=1)


# Print the model summary
model.summary()

# Make predictions using the model
prediction = model.predict(hit_matrix)
#print(np.shape(prediction))
#prediction = tf.clip_by_value(tf.round(prediction),0,201)
#print(prediction[0][0])


# Combine labels for both muon types
# Truth_mup_mum = np.hstack((Truth_elementID_mup, Truth_elementID_mum))

# Calculate the residuals
Res0 = prediction[0] - Truth_elementID_mup
Res1 = prediction[1] - Truth_elementID_mum

print(np.shape(Res0))


#print(np.shape(Res))

# Create 'plots' directory if it doesn't exist
output_dir = "plots_V6"
os.makedirs(output_dir, exist_ok=True)


detectors_order= ['D0U_ele', 'D0Up_ele','D0X_ele','D0Xp_ele','D0V_ele','D0Vp_ele',
            'D2V_ele','D2Vp_ele','D2Xp_ele','D2X_ele','D2U_ele','D2Up_ele',
            'D3pVp_ele','D3pV_ele','D3pXp_ele','D3pX_ele','D3pUp_ele','D3pU_ele',
            'D3mVp_ele','D3mV_ele','D3mXp_ele','D3mX_ele','D3mUp_ele','D3mU_ele',
            'H1B_ele','H1T_ele','H1L_ele','H1R_ele','H2L_ele','H2R_ele',
            'H2B_ele','H2T_ele','H3B_ele','H3T_ele','H4Y1L_ele','H4Y1R_ele','H4Y2L_ele',
            'H4Y2R_ele','H4B_ele','H4T_ele','P1Y1_ele','P1Y2_ele','P1X1_ele','P1X2_ele',
            'P2X1_ele','P2X2_ele','P2Y1_ele','P2Y2_ele']
#print(len(detectors_order))

#removes prop
detectors_order = detectors_order[:40]
num_detectors = len(detectors_order)


positiveMeans = np.zeros(num_detectors)
positiveWidths = np.zeros_like(positiveMeans)

negativeMeans = np.zeros(num_detectors)
negativeWidths = np.zeros_like(negativeMeans)



# Loop through each detector slice, plot residual histograms for positive and negative detectors
for i in range(num_detectors):

    # Slice residuals for the ith detector (positive and negative)
    res_positive = Res0[:, i]         # Positive detectors (first 62 columns)
    res_negative = Res1[:, i]    # Negative detectors (last 62 columns)

    # Calculate mean and standard deviation for positive and negative residuals
    mean_positive = np.mean(res_positive)
    std_positive = np.std(res_positive)
    mean_negative = np.mean(res_negative)
    std_negative = np.std(res_negative)

    positiveMeans[i] = mean_positive
    negativeMeans[i] = mean_negative

    positiveWidths[i] = std_positive
    negativeWidths[i] = std_negative

    print(mean_positive)
    # Create a new figure for each detector
    plt.figure(figsize=(10, 6))
    
    #Note: Not sure what the binning should be....
    # Plot histogram for positive residuals in green
    plt.hist(res_positive, bins=48, color='green', alpha=0.7, label=f'Residuals: Detector {detectors_order[i]} (Positive)')
    
    # Plot histogram for negative residuals in red
    plt.hist(res_negative, bins=48, color='red', alpha=0.7, label=f'Residuals: Detector {detectors_order[i]} (Negative)')
    
    # Add mean and std text for positive residuals
    plt.text(0.95, 0.75, f'Positive\nMean: {mean_positive:.2f}\nStd Dev: {std_positive:.2f}', 
             color='green', ha='right', va='top', transform=plt.gca().transAxes)

    # Add mean and std text for negative residuals
    plt.text(0.95, 0.55, f'Negative\nMean: {mean_negative:.2f}\nStd Dev: {std_negative:.2f}', 
             color='red', ha='right', va='top', transform=plt.gca().transAxes)
    
    # Title and labels
    plt.title(f'Residuals Histogram for Detector {detectors_order[i]}_{i}')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    
    # Show legend
    plt.legend(loc='upper right')
    
    # Save the plot to the 'plots' directory
    plot_filename = os.path.join(output_dir, f'Residuals_Histogram_Detector_{i}_{detectors_order[i]}.png')
    plt.savefig(plot_filename)
    
    # Close the plot to free memory
    plt.close()

# Plot the histogram of total residuals for all detectors
plt.figure(figsize=(10, 6))
res_positive = Res0    
res_negative = Res1  


# Plot histogram for positive residuals in green
plt.hist(res_positive.flatten(), bins=200, color='green', alpha=0.7, label='Residuals: Positive Detectors')

# Plot histogram for negative residuals in red
plt.hist(res_negative.flatten(), bins=200, color='red', alpha=0.7, label='Residuals: Negative Detectors')

plt.axvline(0, color='k', linestyle='dashed', linewidth=1, label='Zero Residual')
plt.title('Total Residual Histogram')
plt.xlabel('Residual Value')
plt.ylabel('Frequency')
plt.legend()
plt.xlim(-10, 10)
plt.grid(True)

# Save the total residual plot
total_residual_filename = os.path.join(output_dir, 'Total_Residual_Histogram.png')
plt.savefig(total_residual_filename)

# Close the plot to free memory
plt.close()


print(len(detectors_order))

# Define subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

# Data and titles for each subplot
data_and_titles = [
    (positiveMeans, "Positive Means"),
    (negativeMeans, "Negative Means"),
    (positiveWidths, "Positive Widths"),
    (negativeWidths, "Negative Widths")
]

# Iterate over the subplots and data
for ax, (data, title) in zip(axs.flat, data_and_titles):
    ax.bar(detectors_order, data, color='blue', alpha=0.7)
    ax.set_title(title)
    ax.set_xticks(range(len(detectors_order)))
    ax.set_xticklabels(detectors_order, rotation=90, fontsize=8)
    ax.set_ylabel('Value')

Mean_and_Widths_filename = os.path.join(output_dir, 'Means_and_Widths.png')
plt.savefig(Mean_and_Widths_filename)
