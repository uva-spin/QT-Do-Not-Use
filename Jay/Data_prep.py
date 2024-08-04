# %%
import numpy as np
import uproot
from numba import njit, prange
import numba

# %%
def read_root_file(root_file):
    print("Reading ROOT file...")
    targettree = uproot.open(root_file+':QA_ana')
    targetevents = len(targettree['n_tracks'].array(library='np'))

    # List of detector names
    detector_names = [
    'D0V_ele', 'D0Vp_ele', 'D0X_ele', 'D0Xp_ele', 'D0U_ele', 'D0Up_ele', 
    'H1L_ele', 'H1R_ele', 'H1B_ele', 'H1T_ele', 'D2V_ele', 'D2Vp_ele', 
    'D2Xp_ele', 'D2X_ele', 'D2U_ele', 'D2Up_ele', 'H2R_ele', 'H2L_ele', 
    'H2B_ele', 'H2T_ele', 'D3mVp_ele', 'D3mV_ele', 'D3mXp_ele', 'D3mX_ele', 
    'D3mUp_ele', 'D3mU_ele', 'D3pVp_ele', 'D3pV_ele', 'D3pXp_ele', 'D3pX_ele', 
    'D3pUp_ele', 'D3pU_ele', 'H3B_ele', 'H3T_ele', 'P1X1_ele', 'H4Y1L_ele', 
    'H4Y1R_ele', 'P1Y1_ele', 'H4Y2L_ele', 'H4Y2R_ele', 'H4T_ele', 'H4B_ele', 
    'P2Y1_ele', 'P2X1_ele', 'P2X2_ele'
    ]

    # Dynamically assign values from targettree to the corresponding variables
    for detector_name in detector_names:
        globals()[detector_name] = targettree[detector_name].array(library='np')
        print(detector_name, globals()[detector_name].shape)  # Verify shape

    # List of detector arrays in the correct order
    detector_arrays = [
        D0V_ele, D0Vp_ele, D0X_ele, D0Xp_ele, D0U_ele, D0Up_ele, H1L_ele, H1R_ele, H1B_ele, H1T_ele,
        D2V_ele, D2Vp_ele, D2Xp_ele, D2X_ele, D2U_ele, D2Up_ele, H2R_ele, H2L_ele, H2B_ele, H2T_ele,
        D3mVp_ele, D3mV_ele, D3mXp_ele, D3mX_ele, D3mUp_ele, D3mU_ele, D3pVp_ele, D3pV_ele, D3pXp_ele, 
        D3pX_ele, D3pUp_ele, D3pU_ele, H3B_ele, H3T_ele,
        P1X1_ele, H4Y1L_ele, H4Y1R_ele, P1Y1_ele, H4Y2L_ele, H4Y2R_ele, H4T_ele, H4B_ele, P2Y1_ele, 
        P2X1_ele, P2X2_ele
    ]

    pid = targettree['pid'].array(library='np')

    print('Reading Done')

    elemID_mup = np.zeros((targetevents, 500), dtype=int)
    elemID_mum = np.zeros((targetevents, 500), dtype=int)
    elemID = np.zeros((targetevents, 500,2), dtype=int)
    print(targetevents)
    for event in range(targetevents):
        number_of_tracks = np.size(pid[event])
        for track in range(number_of_tracks):
            if pid[event][track] > 0:
                for i, detector in enumerate(detector_arrays, start=1):
                    print(f'Assigning {detector[event][track]} to elemID_mup[{event}][{i}]')
                    elemID_mup[event][i] = detector[event][track]

                elemID[event] = np.column_stack((elemID_mup[event], elemID_mum[event]))
            elif pid[event][track] < 0:
                for i, detector in enumerate(detector_arrays, start=1):
                    print(f'Assigning {detector[event][track]} to elemID_mum[{event}][{i}]')
                    elemID_mum[event][i] = detector[event][track]

                elemID[event] = np.column_stack((elemID_mum[event], elemID_mup[event]))
            
        
    #clean elementID
    elemID[:][elemID[:] >= 1000] = 0
    elemID_mum[:][elemID_mum[:] >= 1000] = 0
    elemID_mup[:][elemID_mup[:] >= 1000] = 0

    return elemID_mum, elemID_mup, elemID


# %%
root_file = "rootfiles/DY_Target_100k_080124/merged_trackQA_v2.root"
elemID_mum, elemID_mup, elemID= read_root_file(root_file)

# %%
def Get_Drifts(root_file):
    print("Reading ROOT file...")
    targettree = uproot.open(root_file+':QA_ana')
    targetevents = len(targettree['n_tracks'].array(library='np'))

    # List of detector names
    detector_names = [
    "D0U_drift","D0Up_drift", "D0X_drift", "D0Xp_drift", "D0V_drift", "D0Vp_drift",
    "D2U_drift", "D2Up_drift", "D2X_drift", "D2Xp_drift", "D2V_drift",
    "D2Vp_drift", "D3pU_drift", "D3pUp_drift", "D3pX_drift", "D3pXp_drift",
    "D3pV_drift", "D3pVp_drift", "D3mU_drift", "D3mUp_drift", "D3mX_drift",
    "D3mXp_drift", "D3mV_drift", "D3mVp_drift"
    ]

    # Dynamically assign values from targettree to the corresponding variables
    for detector_name in detector_names:
        globals()[detector_name] = targettree[detector_name].array(library='np')
        print(detector_name, globals()[detector_name].shape)  # Verify shape

    # List of detector arrays in the correct order
    detector_arrays = [
        D0U_drift,D0Up_drift,D0X_drift,D0Xp_drift,D0V_drift,D0Vp_drift,
    D2U_drift,D2Up_drift,D2X_drift,D2Xp_drift,D2V_drift,
    D2Vp_drift, D3pU_drift, D3pUp_drift, D3pX_drift, D3pXp_drift,
    D3pV_drift, D3pVp_drift, D3mU_drift, D3mUp_drift, D3mX_drift,
    D3mXp_drift, D3mV_drift, D3mVp_drift
    ]

    pid = targettree['pid'].array(library='np')

    print('Reading Done')

    drifts_mup = np.zeros((targetevents, 500))
    drifts_mum = np.zeros((targetevents, 500))
    drifts = np.zeros((targetevents, 500,2))

    dc_detid = np.hstack((np.arange(1,7),np.arange(11,17),np.arange(21,33)))

    print(targetevents)
    for event in range(targetevents):
        number_of_tracks = np.size(pid[event])
        for track in range(number_of_tracks):
            if pid[event][track] > 0:
                for i, detector in enumerate(detector_arrays, start=1):

                    if(i in dc_detid):
                        print(f'Assigning {detector[event][track]} to drifts_mup[{event}][{i}]')
                        drifts_mup[event][i] = detector[event][track]
                    else:
                        #print(f'Assigning 0 to drifts_mup[{event}][{i}]')
                        drifts_mup[event][i] = 0
                drifts[event] = np.column_stack((drifts_mup[event], drifts_mum[event]))
            elif pid[event][track] < 0:
                for i, detector in enumerate(detector_arrays, start=1):
                    print(type(detector[event][track]))
                    if(i in dc_detid):
                        print(f'Assigning {detector[event][track]} to drifts_mum[{event}][{i}]')
                        drifts_mum[event][i] = detector[event][track]
                    else:
                        #print(f'Assigning 0 to drifts_mum[{event}][{i}]')
                        drifts_mum[event][i] = 0

                drifts[event] = np.column_stack((drifts_mum[event], drifts_mup[event]))
            
    #clean out high valves    
    drifts[:][drifts[:] >= 1000] = 0
    drifts_mum[:][drifts_mum[:] >= 1000] = 0
    drifts_mup[:][drifts_mup[:] >= 1000] = 0

    return drifts_mum, drifts_mup, drifts


# %%
drifts_mum, drifts_mup, drifts= Get_Drifts(root_file)

# %%
def count_hits(events):
    missing_hits = np.zeros(len(events))
    count = 0
    ecount = 0
    for event in range(len(events)):
        for hit in events[event]:
            if hit[0] != 0 and hit[1] != 0:
                count += 1

        if(count < 32):
            missing_hits[ecount] = event
            ecount += 1
    return missing_hits


# %%
missing_hits = count_hits(elemID)

# %%
def too_many_tracks_events(root_file):
    print("Reading ROOT file...")
    targettree = uproot.open(root_file+':QA_ana')
    targetevents=len(targettree['n_tracks'].array(library='np'))
    n_tracks = targettree['n_tracks'].arrays(library="np")['n_tracks']
    
    bad_event = []

    for i in range(len(n_tracks)):
        if(n_tracks[i] != 2):
            bad_event = np.append(bad_event,i)



    return bad_event

# %%
#root_file = "rootfiles/DY_Target_100k_080124/merged_trackQA_v2.root"
too_many_tracks_event = too_many_tracks_events(root_file).astype(int)

# %%
too_many_tracks_event

# %%
bad_events = np.concatenate((missing_hits[np.where(missing_hits != 0)], too_many_tracks_event))

# %%

def build_hitmatrix(events,drifts,bad_events):


    number_of_events = len(events)
    hit_matrix = np.zeros((number_of_events,47,202))

    for event in range(number_of_events):
        DC_hit = 0
        if(event is not bad_events):
            for detid in range(1,46):
                n_track = np.size(events[event][0])
                if (n_track == 1):
                    hit = events[event][detid]
                    if(hit != 0):
                        hit_matrix[event][detid][hit] = drifts[event][detid]
                else:
                    for track in range(n_track):
                        hit = events[event][detid][track]
                        if(hit != 0):
                            hit_matrix[event][detid][hit] = drifts[event][detid][track]

    return hit_matrix

# %%
hitmatrix = build_hitmatrix(elemID,drifts,bad_events)
hitmatrix_mup = build_hitmatrix(elemID_mup,drifts_mup,bad_events)
hitmatrix_mum = build_hitmatrix(elemID_mum,drifts_mum,bad_events)

# %%
np.savez('Hitmatrices.npz', hitmatrix=hitmatrix, hitmatrix_mup=hitmatrix_mup,hitmatrix_mum=hitmatrix_mum)


