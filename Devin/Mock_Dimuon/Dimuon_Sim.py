import numpy as np
import uproot
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
# from Generate_Background import Background

class Data_Processing:
    ''''
    This program is to process data coming from QA_V2 for SpinQuest and UVA. 
    This program reads in branches from the root file, filters ideal events, and organize data into
    a Hit matrix and Ground Truth labels. The goal is to preprocess the data to be used in a DNN. 
    '''

    def __init__(self, rootfile: str) -> None:
        self.rootfile = rootfile
        print("Reading the Root file")
        self.TTree = uproot.open(self.rootfile + ":QA_ana")

        self.detectors_order = np.array(['D0U_ele', 'D0Up_ele','D0X_ele','D0Xp_ele','D0V_ele','D0Vp_ele',
                                         'NaN','NaN','NaN','NaN','NaN','NaN',
                                         'D2V_ele','D2Vp_ele','D2Xp_ele','D2X_ele','D2U_ele','D2Up_ele',
                                         'D3pVp_ele','D3pV_ele','D3pXp_ele','D3pX_ele','D3pUp_ele','D3pU_ele',
                                         'D3mVp_ele','D3mV_ele','D3mXp_ele','D3mX_ele','D3mUp_ele','D3mU_ele',
                                         'H1B_ele','H1T_ele','H1L_ele','H1R_ele','H2L_ele','H2R_ele',
                                         'H2B_ele','H2T_ele','H3B_ele','H3T_ele','H4Y1L_ele','H4Y1R_ele',
                                         'H4Y2L_ele','H4Y2R_ele','H4B_ele','H4T_ele','P1Y1_ele','P1Y2_ele',
                                         'P1X1_ele','P1X2_ele','P2X1_ele','P2X2_ele','P2Y1_ele','P2Y2_ele',
                                         'NaN','NaN','NaN','NaN','NaN','NaN','NaN','NaN'])

        self.drift_order = np.array(['D0U_drift', 'D0Up_drift','D0X_drift','D0Xp_drift','D0V_drift','D0Vp_drift',
                                     'NaN','NaN','NaN','NaN','NaN','NaN',
                                     'D2V_drift','D2Vp_drift','D2Xp_drift','D2X_drift','D2U_drift','D2Up_drift',
                                     'D3pVp_drift','D3pV_drift','D3pXp_drift','D3pX_drift','D3pUp_drift','D3pU_drift',
                                     'D3mVp_drift','D3mV_drift','D3mXp_drift','D3mX_drift','D3mUp_drift','D3mU_drift'])
        

        self.useful_info = np.array(['n_tracks', 'elementID','detectorID','pid'])

        # Remove 'NaN' from arrays to only load valid branch names
        self.detectors = self.detectors_order[self.detectors_order != 'NaN']
        self.drifts= self.drift_order[self.drift_order != 'NaN']

        # Load branches as a dictionary of numpy arrays
        self.data = self.load_branches()
        self.num_events = self.data


    def load_branches(self):
        # Combine the detectors and drift branches into a single list
        branch_names = np.concatenate((self.detectors, self.drifts, self.useful_info))

        # Use uproot to read all specified branches at once
        arrays = self.TTree.arrays(branch_names, library="np")

        return arrays




    def get_num_events(self):
        #Obtains number of events in file.
        count = len(self.data['n_tracks'])
        print(f"We have {count} events in this file!")
        return count   
        
    def get_branch_info(self,branch: str,event: int):
        #This function reads in the data when given a branch and event. 
        #Returns Nothing if the branch isn't used. Unused detectors are labeled 'NAN'
        if(branch == 'NaN'):
            return None
        else:
            detector = self.TTree[branch].array(library='np')
            return detector[event]
        
    def find_ideal_events(self,event: int):
        '''
        This, given an event, checks if the event is an ideal event.
        An Ideal event has 6 hits in each station except for st3 which has 6 in 
        the + or - station. 
        This can be tuned if needed.
        Returns True or False.
        '''
        #Find # of tracks
        n_track =  self.data['n_tracks'][event]
        detectorID = self.data['detectorID'][event]
        detectorID = detectorID[detectorID <= 31]
        #Count must = 6 per station per track
        hits_per_station = n_track*6

        #Counts in each station
        st1_count = len(np.where(detectorID <= 6)[0])
        st2_count = len(np.where((detectorID >= 13) & (detectorID <= 18))[0])
        st3p_count = len(np.where((detectorID >= 19) & (detectorID <= 24))[0])
        st3m_count = len(np.where((detectorID >= 25) & (detectorID <= 31))[0])

        #print(st1_count,st2_count,st3p_count,st3m_count)
        #Case per station
        good_event = False

        if hits_per_station == st1_count:
            good_event = True
        else:
            good_event = False

        if hits_per_station == st2_count:
            good_event = True
        else:
            good_event = False

        hits_per_station = 6
        if hits_per_station == st3p_count:
            good_event = True
        elif hits_per_station == st3m_count:
            good_event = True
        else:
            #print("Not an Ideal event!")
            good_event = False

        return good_event
    

    def make_Hitmatrix(self,ideal_events):
        '''
        Given a list of events, creates the hitmatrix. For each detector, the branch information is 
        collected. The Drift distance is also collected. 
        This information is stored in a hitmatrix, and truth arrays. 
        '''
        
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
          
            #Each Driftchamber has a drift distance. 
        drift_order =    np.array(['D0U_drift', 'D0Up_drift','D0X_drift','D0Xp_drift','D0V_drift','D0Vp_drift',
                          'NaN','NaN','NaN','NaN','NaN','NaN',
                          'D2V_drift','D2Vp_drift','D2Xp_drift','D2X_drift','D2U_drift','D2Up_drift',
                          'D3pVp_drift','D3pV_drift','D3pXp_drift','D3pX_drift','D3pUp_drift','D3pU_drift',
                          'D3mVp_drift','D3mV_drift','D3mXp_drift','D3mX_drift','D3mUp_drift','D3mU_drift'])
        
 
        print("Creating the hitmatrix")

        detectorID_order = np.arange(1,63)
        
        num_events = len(ideal_events)
        hit_matrix = np.zeros((num_events,62,201),dtype=np.bool_)
        Truth_elementID_mup = np.zeros((num_events,62),dtype=np.uint8)
        Truth_elementID_mum = np.zeros((num_events,62),dtype=np.uint8)
        Truth_values_drift_mup  = np.zeros((num_events,62))
        Truth_values_drift_mum  = np.zeros((num_events,62))

        # num_events = len(ideal_events)
        # hit_matrix_mup = np.zeros((62, 201), dtype=np.int32)
        # hit_matrix_mum = np.zeros((62, 201), dtype=np.int32)

        # for i, event in enumerate(ideal_events):
        #     event = int(event)
        #     pid = self.data['pid'][event]
        #     n_track = self.data['n_tracks'][event]

        #     for j, detector in enumerate(detectors_order):
        #         if detector != 'NaN':
        #             hit_info = self.data[detector][event]
        #             drift_varible = drift_order[j] if j < len(drift_order) else None

        #             for track in range(n_track):
        #                 hit = hit_info[track]
        #                 if hit < 201:  # Remove high-voltage hits
        #                     if pid[track] > 0:  # Positive muon
        #                         hit_matrix_mup[j, hit] += 1
        #                     else:  # Negative muon
        #                         hit_matrix_mum[j, hit] += 1

        # return hit_matrix_mup, hit_matrix_mum
        
        event_index = 0
        for i, event in enumerate(ideal_events):
            #loops over every event in ideal event list. Event index, keeps track for the Truth Arrays.
            event = int(event)
            pid = self.data['pid'][event]
            n_track = self.data['n_tracks'][event]


            for j, detector in enumerate(detectors_order):                
                    #print(detector,det_index)
                for track in range(n_track):
                #Loops over each track in the event        
                #These indices are to keep track of detID and the hitmatrix.

                    if(j < 30):
                        #Checks if we are in the Drift chamber.
                        if( detector != 'NaN'):
                            hit_info = self.data[detector][event]
                        
                            #Check if the detector is not NAN
                            hit = hit_info[track]

                            drift_varible = drift_order[j]
                            drift_info = drift = self.data[drift_varible][event]
                            drift = drift_info[track]

                            if(pid[track] > 0):
                                #Check for the particle ID
                                if(hit < 201):
                                    #Removes high voltage hits
                                    Truth_elementID_mup[i,j] = hit
                                    hit_matrix[i,j,hit] = True
                                    Truth_values_drift_mup[i,j] = drift
                          
                            else:
                                #Negative muon
                                if(hit < 201):
                                    #Removes high voltage hits
                                    Truth_elementID_mum[i,j]= hit
                                    hit_matrix[i,j,hit] = True
                                    Truth_values_drift_mum[i,j] = drift

                    else:
                        #If not in the drift chamber drift distance = 1
                        if( detector != 'NaN'):
                            hit_info = self.data[detector][event]
                        
                            #Check if the detector is not NAN
                            hit = hit_info[track]
                            if(pid[track] > 0):
                                #Checks the particle ID.
                                if(hit < 201):
                                    #Removes high voltage hits.
                                    Truth_elementID_mup[i,j] = hit
                                    hit_matrix[i,j,hit] = True
                            else:
                                #Negative muon
                                if(hit < 201):
                                    #Gets rid of high voltage hits
                                    Truth_elementID_mum[i,j] = hit
                                    hit_matrix[i,j,hit] = True
        return Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix


# if __name__ == "__main__":
#     # Load your root file and initialize the Data_Processing object
#     start = time()
#     # Reads in root file
#     root_file = "/home/devin/Documents/Big_Data/Dimuon_Mock/Dimuon_target_100K.root"
#     dp = Data_Processing(root_file)

#     # Set number of events
#     num_events = dp.get_num_events()

#     # Create an array of ideal events
#     ideal_events = np.zeros(num_events)
#     for event in range(num_events):
#         good_event = dp.find_ideal_events(event)
#         if good_event:
#             ideal_events[event] = event
#     ideal_events = ideal_events[ideal_events != 0]

#     # Define how many events to sum up
#     n = 50000  # Change this to the number of events you want to process
#     selected_events = ideal_events[:n]

#     # Generate hit matrices
#     hit_matrix_mup, hit_matrix_mum = dp.make_Hitmatrix(selected_events)

#     # Choose plotting mode
#     plot_mode = "hits"  # Options: "hits" or "heatmap"

#     # Create a figure and axis
#     fig, ax = plt.subplots(figsize=(16, 8))

#     if plot_mode == "heatmap":
#         # Heatmap mode
#         ax.imshow(hit_matrix_mup.T, cmap='Reds', interpolation='none', origin='upper')
#         ax.imshow(hit_matrix_mum.T, cmap='Blues', interpolation='none', alpha=0.7, origin='upper')
#     elif plot_mode == "hits":
#         # Hits mode: Plot individual points
#         mup_y, mup_x = np.where(hit_matrix_mup.T > 0)
#         mum_y, mum_x = np.where(hit_matrix_mum.T > 0)
#         ax.scatter(mup_x, mup_y, color='blue', label='Muon Plus', marker='_')
#         ax.scatter(mum_x, mum_y, color='red', label='Muon Minus', marker = '_')

#     # Set axis labels
#     ax.set_xlabel("Detector ID")
#     ax.set_ylabel("Element ID")

#     # Flip the y-axis
#     ax.invert_yaxis()

#     # Adjust aspect ratio for better visibility
#     ax.set_aspect(0.1)

#     # Add a grid for clarity
#     ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

#     # Add a legend for hits mode
#     if plot_mode == "hits":
#         ax.legend(loc='upper right')

#     # Show the plot
#     plt.title("Overlay of Tracks")
#     plt.show()




# Example
start = time()
#Reads in root file
root_file = "/home/devin/Documents/Big_Data/Dimuon_Mock/Dimuon_target_100K.root"
data_processor = Data_Processing(root_file)

#Set number of events
num_events = data_processor.get_num_events()

ideal_events = np.zeros(num_events)
# Create an array of ideal events
for event in range(num_events):
    good_event = data_processor.find_ideal_events(event)
    if good_event:
        ideal_events[event] = event
ideal_events = ideal_events[ideal_events != 0]

print(f"There are this many ideal events: {len(ideal_events)}")



#Make hitmatrix
Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix = data_processor.make_Hitmatrix(ideal_events)

np.savez('/home/devin/Documents/Big_Data/Dimuon_Mock/Hit_Info.npz', Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum,hit_matrix,ideal_events)

#print(ideal_events)


stop = time()
print(stop-start)


# Initialize the plot for combining all scatter plots
plt.figure(figsize=(10, 6))
detID = np.arange(1,63)
n = 1000
# Loop over all the events with a progress bar using tqdm
for event in tqdm(ideal_events[:n], desc="Processing Events"):
    event = int(event)

    # Get the Truth event index for the current event
    Truth_event = np.where(ideal_events == event)


    # Scatter plots for Mup and Mum, overlaying them onto the same figure
    plt.scatter(detID, Truth_elementID_mup[Truth_event], marker='_', color='r', alpha=0.5)
    plt.scatter(detID, Truth_elementID_mum[Truth_event], marker='_', color='g', alpha=0.5)

# Add labels, title, and formatting
plt.xlim(0, 64)
plt.ylim(0, 201)
plt.title("Combined Truth Events")
plt.xlabel("DetectorID")
plt.ylabel("ElementID")
plt.legend(['Mup', 'Mum'])

# Save and show the combined plot
plt.savefig(r"/home/devin/Documents/Big_Data/Dimuon_Mock/Combined_Truth_Events.jpeg")
plt.show()

stop = time()
print(f"Processing time: {stop - start} seconds")