import numpy as np
import uproot
from numba import njit, prange
import numba

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
    elemID = np.zeros((targetevents, 500), dtype=int)
    print(targetevents)
    for event in range(targetevents):
        number_of_tracks = np.size(pid[event])
        for track in range(number_of_tracks):
            if pid[event][track] > 0:
                for i, detector in enumerate(detector_arrays, start=1):
                    print(f'Assigning {detector[event][track]} to elemID_mup[{event}][{i}]')
                    elemID_mup[event][i] = detector[event][track]
            elif pid[event][track] < 0:
                for i, detector in enumerate(detector_arrays, start=1):
                    print(f'Assigning {detector[event][track]} to elemID_mum[{event}][{i}]')
                    elemID_mum[event][i] = detector[event][track]

    return elemID_mum, elemID_mup

@numba.jit(nopython=True)
def clean(events):
    for j in range(len(events)):
            for i in range(500):
                if(events[j][i]>1000):
                    events[j][i]=0
    return events


root_file = "merged_trackQA_v2.root"
elemID_mum, elemID_mup = read_root_file(root_file)
elemID_mum = clean(elemID_mum).astype(int)
elemID_mup = clean(elemID_mup).astype(int)