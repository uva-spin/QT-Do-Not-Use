import numpy as np
import uproot
import matplotlib.pyplot as plt

class Data_Processing:
    ''''
    This program is to process data coming from QA_V2 for SpinQuest and UVA. 
    This program reads in branches from the root file, filters ideal events, and organize data into
    a Hit matrix and Ground Truth labels. The goal is to preprocess the data to be used in a DNN. 
    '''

    def __init__(self,rootfile: str) -> None:
        self.rootfile = rootfile
        print("Reading the Root file")
        self.TTree = uproot.open(self.rootfile+":QA_ana")


    def get_num_events(self):
        #Obtains number of events in file.
        count = self.TTree['n_tracks'].arrays(library='np')
        count = len(count['n_tracks'])  
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
        n_track =  data_processor.get_branch_info('n_tracks',event)

        detectorID = np.array(data_processor.get_branch_info('detectorID',event))
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
        hit_matrix = np.zeros((num_events,62,201))
        Truth_elementID_mup = np.zeros((num_events,62))
        Truth_elementID_mum = np.zeros((num_events,62))
        Truth_values_drift_mup  = np.zeros((num_events,62))
        Truth_values_drift_mum  = np.zeros((num_events,62))
        
        event_index = 0
        for event in ideal_events:
            #loops over every event in ideal event list. Event index, keeps track for the Truth Arrays.
            event = int(event)
            pid = data_processor.get_branch_info('pid',event)
            n_track =  data_processor.get_branch_info('n_tracks',event)

            for track in range(n_track):
                #Loops over each track in the event        
                det_index = 0
                hit_index = 0
                #These indices are to keep track of detID and the hitmatrix.
                for detector in detectors_order:                
                    #print(detector,det_index)

                    if(det_index < 30):
                        #Checks if we are in the Drift chamber.
                        hit_info = data_processor.get_branch_info(detector,event)
                        if hit_info is not None:
                            #Check if the detector is not NAN
                            hit = hit_info[track]

                            drift_varible = drift_order[det_index]
                            drift_info = data_processor.get_branch_info(drift_varible,event)
                            
                            drift = drift_info[track]

                            if(pid[track] > 0):
                                #Check for the particle ID
                                if(hit < 202):
                                    #Removes high voltage hits
                                    Truth_elementID_mup[event_index,det_index] = hit
                                    hit_matrix[hit_index,det_index,hit] = drift
                                    Truth_values_drift_mup[event_index,det_index] = drift
                          
                            else:
                                #Negative muon
                                if(hit < 202):
                                    #Removes high voltage hits
                                    Truth_elementID_mum[event_index,det_index]= hit
                                    hit_matrix[hit_index,det_index,hit] = drift
                                    Truth_values_drift_mum[event_index,det_index] = drift

                    else:
                        #If not in the drift chamber drift distance = 1
                        hit_info = data_processor.get_branch_info(detector,event)
                        if hit_info is not None:
                            #Check if the detector is not NAN
                            hit = hit_info[track]
                            if(pid[track] > 0):
                                #Checks the particle ID.
                                if(hit < 202):
                                    #Removes high voltage hits.
                                    Truth_elementID_mup[event_index,det_index] = hit
                                    hit_matrix[hit_index,det_index,hit] = 1
                            else:
                                #Negative muon
                                if(hit < 202):
                                    #Gets rid of high voltage hits
                                    Truth_elementID_mum[event_index,det_index] = hit
                                    hit_matrix[hit_index,det_index,hit] = 1
                    det_index += 1
                hit_index += 1
            event_index += 1
        return Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix


#Reads in root file
root_file = "/Users/jay/Documents/Research/machine_learning/rootfiles/DY_Target_27M_083124/merged_trackQA_v2.root"
data_processor = Data_Processing(root_file)
#Set number of events
num_events = data_processor.get_num_events()

ideal_events = []
#Create an array of ideal events
for event in range(num_events):
    good_event = data_processor.find_ideal_events(event)
    if good_event:
        ideal_events = np.append(ideal_events,event)

print(f"There are this many ideal events: {len(ideal_events)}")
print(ideal_events)


#Make hitmatrix
Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum, hit_matrix = data_processor.make_Hitmatrix(ideal_events)

np.savez('Hit_Info.npz', Truth_elementID_mup, Truth_elementID_mum, Truth_values_drift_mup, Truth_values_drift_mum,hit_matrix,ideal_events)

# print("Testing")

# event = int(ideal_events[0])
# print(event)

# elementID =  data_processor.get_branch_info('elementID',event)
# detectorID = data_processor.get_branch_info('detectorID',event)

# Truth_event = np.where(ideal_events == event)[0][0]

# detID = np.arange(1,63)
# plt.scatter(detID,Truth_elementID_mup[Truth_event],marker='o',color='r')
# plt.scatter(detID,Truth_elementID_mum[Truth_event],marker='d',color='g')
# plt.scatter(detectorID,elementID,marker='+',color='k')

# plt.xlim(0,64)
# plt.ylim(0,201)
# plt.title("Truth Event")
# plt.xlabel("DetectorID")
# plt.ylabel("ElementID")
# plt.show()
