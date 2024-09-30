
from dataProcess import Data_Processing
import numpy as np
from time import time
from Merge_npz import combine_npz_files

start = time()
root_file = "rootfiles/DY_Target_27M_083124/merged_trackQA_v2.root"
data_processor = Data_Processing(root_file)

num_events = data_processor.get_num_events()

ideal_events = np.zeros(num_events)
# Create an array of ideal events
for event in range(num_events):
    good_event = data_processor.find_ideal_events(event)
    if good_event:
        ideal_events[event] = event
ideal_events = ideal_events[ideal_events != 0]

print(f"There are this many ideal events: {len(ideal_events)}")


def batch_generator(data, batch_size):
    """Yield successive batches from the data."""
    for i in range(0, len(data), batch_size):
        #Allows to return a batch without exiting the loop
        yield data[i:i + batch_size]

# Use a generator to avoid loading all events into memory at once
batch_size = 100000
batch_gen = batch_generator(ideal_events, batch_size)

# Loop through the generator
for i, batch_events in enumerate(batch_gen, 1):
    # Process the batch with make_Hitmatrix
    Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix = data_processor.make_Hitmatrix(batch_events)

    # Save the batch results to the npz file using compression
    with open(f'Hit_Data/Hit_Info_batch_{i}.npz', 'wb') as f:
        np.savez_compressed(f, Truth_elementID_mup=Truth_elementID_mup, 
                                  Truth_elementID_mum=Truth_elementID_mum, 
                                  Truth_values_drift_mup=Truth_values_drift_mup, 
                                  Truth_values_drift_mum=Truth_values_drift_mum, 
                                  hit_matrix=hit_matrix, 
                                  ideal_events=batch_events)

    print(f"Batch {i} saved with {len(batch_events)} events.")



#Merge and delete files using merge_npz.py:
# Define the folder path and output file name
folder_path = 'Hit_Data'
output_file = 'Hit_Info.npz'
combine_npz_files(folder_path, output_file)

stop = time()

print(f"This code took this long to complete: {stop-start}")



#Make hitmatrix
#Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix = data_processor.make_Hitmatrix(ideal_events)

#np.savez('Hit_Info.npz', Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum,hit_matrix,ideal_events)

# #print(ideal_events)


# stop = time()
# print(stop-start)


# event = int(ideal_events[20])
# print(f"event is: {event}")

# elementID =  data_processor.get_branch_info('elementID',event)
# detectorID = data_processor.get_branch_info('detectorID',event)

# index = np.where((detectorID >= 19) & (detectorID <= 24))
# print(detectorID[index])
# print(elementID[index])

# Truth_event = np.where(ideal_events == event)[0][0]
# print(f"event is: {ideal_events[Truth_event]}")

# detID = np.arange(1,63)
# import matplotlib.pyplot as plt

# plt.scatter(detID,Truth_elementID_mup[Truth_event],marker='o',color='r')
# plt.scatter(detID,Truth_elementID_mum[Truth_event],marker='d',color='g')
# plt.scatter(detectorID,elementID,marker='+',color='k')

# plt.xlim(0,64)
# plt.ylim(0,201)
# plt.title("Truth Event")
# plt.xlabel("DetectorID")
# plt.ylabel("ElementID")
# plt.show()
