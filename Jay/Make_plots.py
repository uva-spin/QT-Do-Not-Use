import uproot
import numpy as np
import matplotlib.pyplot as plt

from dataProcess import Data_Processing

with np.load('Hit_Info.npz') as data:
	print(data.files)
	Truth_elementID_mup = data['Truth_elementID_mup']
	Truth_elementID_mum = data['Truth_elementID_mum']
	Truth_values_drift_mup = data['Truth_values_drift_mup']
	Truth_values_drift_mum = data['Truth_values_drift_mum']
	hit_matrix = data['hit_matrix']
	ideal_events = data['ideal_events']
	
	
root_file = "/Users/jay/Documents/Research/machine_learning/rootfiles/DY_Target_27M_083124/merged_trackQA_v2.root"
data_processor = Data_Processing(root_file)



event = int(ideal_events[200])
print(f"event is: {event}")

elementID =  data_processor.get_branch_info('elementID',event)
detectorID = data_processor.get_branch_info('detectorID',event)

index = np.where((detectorID >= 19) & (detectorID <= 24))
print(detectorID[index])
print(elementID[index])

Truth_event = np.where(ideal_events == event)[0][0]
print(f"event is: {ideal_events[Truth_event]}")

detID = np.arange(1,63)
plt.scatter(detID,Truth_elementID_mup[Truth_event],marker='o',color='r')
plt.scatter(detID,Truth_elementID_mum[Truth_event],marker='d',color='g')
plt.scatter(detectorID,elementID,marker='+',color='k')

plt.xlim(0,64)
plt.ylim(0,201)
plt.title("Truth Event")
plt.xlabel("DetectorID")
plt.ylabel("ElementID")
plt.show()
