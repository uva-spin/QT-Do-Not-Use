import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the model using Keras API
model = tf.keras.models.load_model("/Users/jay/Documents/Research/machine_learning/Model_V1/Version_1")

# Load data from the .npz file
with np.load('/Users/jay/Documents/Research/machine_learning/Eval_Data/Hit_Data/Hit_Info_DY_Target_5M_batch_1.npz') as data:
    hit_matrix = data['hit_matrix']  
    Truth_elementID_mup = data['Truth_elementID_mup']  
    Truth_elementID_mum = data['Truth_elementID_mum']  

# Print the model summary
model.summary()

# Make predictions using the model
prediction = model.predict(hit_matrix)

# Combine labels for both muon types
Truth_mup_mum = np.hstack((Truth_elementID_mup, Truth_elementID_mum))

# Calculate the residuals
Res = prediction - Truth_mup_mum
#print(np.shape(Res))

# Create 'plots' directory if it doesn't exist
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Number of detectors (assuming the shape of each detector data is 62)
num_detectors = 62

detectors_order= ['D0U_ele', 'D0Up_ele','D0X_ele','D0Xp_ele','D0V_ele','D0Vp_ele',
            'NaN','NaN','NaN','NaN','NaN','NaN',
            'D2V_ele','D2Vp_ele','D2Xp_ele','D2X_ele','D2U_ele','D2Up_ele',
            'D3pVp_ele','D3pV_ele','D3pXp_ele','D3pX_ele','D3pUp_ele','D3pU_ele',
            'D3mVp_ele','D3mV_ele','D3mXp_ele','D3mX_ele','D3mUp_ele','D3mU_ele',
            'H1B_ele','H1T_ele','H1L_ele','H1R_ele','H2L_ele','H2R_ele',
            'H2B_ele','H2T_ele','H3B_ele','H3T_ele','H4Y1L_ele','H4Y1R_ele','H4Y2L_ele',
            'H4Y2R_ele','H4B_ele','H4T_ele','P1Y1_ele','P1Y2_ele','P1X1_ele','P1X2_ele',
            'P2X1_ele','P2X2_ele','P2Y1_ele','P2Y2_ele',
            'NaN','NaN','NaN','NaN','NaN','NaN','NaN','NaN']

print(len(detectors_order))

# Loop through each detector slice, plot residual histograms for positive and negative detectors
for i in range(num_detectors):
    # Slice residuals for the ith detector (positive and negative)
    res_positive = Res[:, i]         # Positive detectors (first 62 columns)
    res_negative = Res[:, i + 62]    # Negative detectors (last 62 columns)

    # Create a new figure for each detector
    plt.figure(figsize=(10, 6))
    
    #Note: Not sure what the binning should be....
    # Plot histogram for positive residuals in green
    plt.hist(res_positive, bins=50, color='green', alpha=0.7, label=f'Residuals: Detector {detectors_order[i]} (Positive)')
    
    # Plot histogram for negative residuals in red
    plt.hist(res_negative, bins=50, color='red', alpha=0.7, label=f'Residuals: Detector {detectors_order[i]} (Negative)')
    
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
res_positive = Res[:, :62]         # Positive detectors (first 62 columns)
res_negative = Res[:, -62:]        # Negative detectors (last 62 columns)

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


