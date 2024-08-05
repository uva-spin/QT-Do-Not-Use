import numpy as np
import uproot
from numba import njit, prange
import random

#These are useful variables.
kin_means = np.array([2,0,35,-2,0,35])
kin_stds = np.array([0.6,1.2,10,0.6,1.2,10])

vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

means = np.concatenate((kin_means,vertex_means))
stds = np.concatenate((kin_stds,vertex_stds))


max_ele = [200, 200, 168, 168, 200, 200, 128, 128,  112,  112, 128, 128, 134, 134, 
           112, 112, 134, 134,  20,  20,  16,  16,  16,  16,  16,  16,
        72,  72,  72,  72,  72,  72,  72,  72, 200, 200, 168, 168, 200, 200,
        128, 128,  112,  112, 128, 128, 134, 134, 112, 112, 134, 134,
        20,  20,  16,  16,  16,  16,  16,  16,  72,  72,  72,  72,  72,
        72,  72,  72]


#This Function takes the Track_QA_v2 format Root files and creates arrays for positive and negative element ids, drift, as well as kinematics.
def read_root_file(root_file):
    print("Reading ROOT file...")
    targettree = uproot.open(root_file+':QA_ana')
    targetevents = len(targettree['n_tracks'].array(library='np'))

    # List of kinematic variables
    kinematic = ['gpx', 'gpy', 'gpz', 'gvx', 'gvy', 'gvz']

    # List of drift variables
    drift = ['D0U_drift', 'D0Up_drift', 'D0X_drift', 'D0Xp_drift', 'D0V_drift', 'D0Vp_drift',
             'D2U_drift', 'D2Up_drift', 'D2X_drift', 'D2Xp_drift', 'D2V_drift', 'D2Vp_drift',
             'D3pU_drift', 'D3pUp_drift', 'D3pX_drift', 'D3pXp_drift', 'D3pV_drift', 'D3pVp_drift',
             'D3mU_drift', 'D3mUp_drift', 'D3mX_drift', 'D3mXp_drift', 'D3mV_drift', 'D3mVp_drift']

    # List of detector names excluding kinematic and drift variables
    detectors = [
        'D0U', 'D0Up', 'D0X', 'D0Xp', 'D0V', 'D0Vp',
        'D2U', 'D2Up', 'D2X', 'D2Xp', 'D2V', 'D2Vp',
        'D3pU', 'D3pUp', 'D3pX', 'D3pXp', 'D3pV', 'D3pVp',
        'D3mU', 'D3mUp', 'D3mX', 'D3mXp', 'D3mV', 'D3mVp',
        'H1B', 'H1T', 'H1L', 'H1R',
        'H2L', 'H2R', 'H2B', 'H2T',
        'H3B', 'H3T',
        'H4Y1L', 'H4Y1R', 'H4Y2L', 'H4Y2R', 'H4B', 'H4T',
        'P1Y1', 'P1Y2', 'P1X1', 'P1X2',
        'P2X1', 'P2X2', 'P2Y1', 'P2Y2'
    ]

    # Read arrays from ROOT file
    arrays = {det: targettree[det + '_ele'].array(library='np') for det in detectors}
    arrays.update({kin: targettree[kin].array(library='np') for kin in kinematic})
    arrays.update({dft: targettree[dft].array(library='np') for dft in drift})
    
    print('Done')

    # Initialize event and kinematics arrays
    pos_events = np.zeros((targetevents, 54))
    pos_drift = np.zeros((targetevents, 24))
    pos_kinematics = np.zeros((targetevents, 6))
    neg_events = np.zeros((targetevents, 54))
    neg_drift = np.zeros((targetevents, 24))
    neg_kinematics = np.zeros((targetevents, 6))

    print("Reading events...")
    for j in range(targetevents):
        first = arrays['pid'][j][0]
        pos, neg = (0, 1) if first > 0 else (1, 0)

        # Assign kinematic values
        for i, kin in enumerate(kinematic):
            pos_kinematics[j][i] = arrays[kin][j][pos]
            neg_kinematics[j][i] = arrays[kin][j][neg]

        # Assign detector values to pos_events and neg_events
        for i, det in enumerate(detectors):
            pos_events[j][i] = arrays[det][j][pos]
            neg_events[j][i] = arrays[det][j][neg]

        # Assign drift values to pos_drift and neg_drift
        for i, dft in enumerate(drift):
            pos_drift[j][i] = arrays[dft][j][pos]
            neg_drift[j][i] = arrays[dft][j][neg]

    print("Finished reading events")
    return pos_events, pos_drift, pos_kinematics, neg_events, neg_drift, neg_kinematics

#This function is used to remove the high values that Track_QA_v2 uses to initialize arrays.
@njit(parallel=True)
def clean(events):
    for j in prange(len(events)):
        for i in prange(54):
            if(events[j][i]>1000):
                events[j][i]=0
    return events


#This function is used to inject NIM3 partial tracks and random hits into the hit matrix.
@njit(parallel=True) ### Trying this with parallel = True - Devin
def hit_matrix(detectorid,elementid,drifttime,hits,drift,station): #Convert into hit matrices
    for j in range (len(detectorid)):
        rand = random.random()
        #St 1
        if(station==1) and (rand<0.85):
            if ((detectorid[j]<7) or (detectorid[j]>30)) and (detectorid[j]<35):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 2
        elif(station==2):
            if (detectorid[j]>12 and (detectorid[j]<19)) or ((detectorid[j]>34) and (detectorid[j]<39)):
                if((detectorid[j]<15) and (rand<0.76)) or ((detectorid[j]>14) and (rand<0.86)) or (detectorid[j]==17):
                    hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                    drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 3
        elif(station==3) and (rand<0.8):
            if (detectorid[j]>18 and (detectorid[j]<31)) or ((detectorid[j]==39) or (detectorid[j]==40)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 4
        elif(station==4):
            if ((detectorid[j]>39) and (detectorid[j]<55)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
    return hits,drift

#This function builds the realistic background of messy hit matrix events. This can be modified to create different background configurations
def build_background(n_events):
    filelist=['output_part1.root:tree_nim3','output_part2.root:tree_nim3','output_part3.root:tree_nim3',
             'output_part4.root:tree_nim3','output_part5.root:tree_nim3','output_part6.root:tree_nim3',
             'output_part7.root:tree_nim3','output_part8.root:tree_nim3','output_part9.root:tree_nim3']
    targettree = uproot.open("/project/ptgroup/QTracker_Training/NIM3/"+random.choice(filelist))
    detectorid_nim3=targettree["det_id"].arrays(library="np")["det_id"]
    elementid_nim3=targettree["ele_id"].arrays(library="np")["ele_id"]
    driftdistance_nim3=targettree["drift_dist"].arrays(library="np")["drift_dist"]
    hits = np.zeros((n_events,54,201))
    drift = np.zeros((n_events,54,201))
    for n in range (n_events): #Create NIM3 events
        g=random.choice([1,2,3,4,5,6])#Creates realistic occupancies for E906 FPGA-1 events. 
        for m in range(g):
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],1)
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],2)
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],3)
            i=random.randrange(len(detectorid_nim3))
            hits[n],drift[n]=hit_matrix(detectorid_nim3[i],elementid_nim3[i],driftdistance_nim3[i],hits[n],drift[n],4)
    del detectorid_nim3, elementid_nim3,driftdistance_nim3
    return hits, drift

# Function to evaluate the Track Finder neural network.
@njit(parallel=True)
def evaluate_finder(testin, testdrift, predictions):
    # The function constructs inputs for the neural network model based on test data
    # and predictions, processing each event in parallel for efficiency.
    reco_in = np.zeros((len(testin), 68, 3))
    
    def process_entry(i, dummy, j_offset):
        j = dummy if dummy <= 5 else dummy + 6
        if dummy > 11:
            if predictions[i][12+j_offset] > 0:
                j = dummy + 6
            elif predictions[i][12+j_offset] < 0:
                j = dummy + 12

        if dummy > 17:
            j = 2 * (dummy - 18) + 30 if predictions[i][2 * (dummy - 18) + 30 + j_offset] > 0 else 2 * (dummy - 18) + 31

        if dummy > 25:
            j = dummy + 20

        k = abs(predictions[i][dummy + j_offset])
        sign = k / predictions[i][dummy + j_offset] if k > 0 else 1
        if(dummy<6):window=15
        elif(dummy<12):window=5
        elif(dummy<18):window=5
        elif(dummy<26):window=1
        else:window=3
        k_sum = np.sum(testin[i][j][k - window:k + window-1])
        if k_sum > 0 and ((dummy < 18) or (dummy > 25)):
            k_temp = k
            n = 1
            while testin[i][j][k - 1] == 0:
                k_temp += n
                n = -n * (abs(n) + 1) / abs(n)
                if 0 <= k_temp < 201:
                    k = int(k_temp)

        reco_in[i][dummy + j_offset][0] = sign * k
        reco_in[i][dummy + j_offset][1] = testdrift[i][j][k - 1]
        if(testin[i][j][k - 1]==1):
            reco_in[i][dummy + j_offset][2]=1

    for i in prange(predictions.shape[0]):
        for dummy in prange(34):
            process_entry(i, dummy, 0)
        
        for dummy in prange(34):
            process_entry(i, dummy, 34)      

    return reco_in

# Drift chamber mismatch calculation
@njit(parallel=True)
def calc_mismatches(track):
    results = []
    for pos_slice, neg_slice in [(slice(0, 6), slice(34, 40)), (slice(6, 12), slice(40, 46)), (slice(12, 18), slice(46, 52))]:
        results.extend([
            np.sum(abs(track[:, pos_slice, ::2, 0] - track[:, pos_slice, 1::2, 0]) > 1, axis=1),
            np.sum(abs(track[:, neg_slice, ::2, 0] - track[:, neg_slice, 1::2, 0]) > 1, axis=1),
            np.sum(abs(track[:, pos_slice, :, 2]) == 0, axis=1),
            np.sum(abs(track[:, neg_slice, :, 2]) == 0, axis=1)
        ])
    return np.array(results)
        