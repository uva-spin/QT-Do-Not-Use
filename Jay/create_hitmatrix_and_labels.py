
import numpy as np
import uproot
from numba import njit, prange


def read_root_file(root_file):
    print("reading root file")
    TTree = uproot.open(root_file+":QA_ana")
 

        
    variables = [
        'n_tracks', 'pid', 'gvx', 'gvy', 'gvz', 'gpx', 'gpy', 'gpz', 'gpt', 'geta', 'gphi',
        'sq_px_st1', 'sq_py_st1', 'sq_pz_st1', 'sq_px_st2', 'sq_py_st2', 'sq_pz_st2',
        'sq_px_st3', 'sq_py_st3', 'sq_pz_st3', 'nhits_track', 'H1B_ele', 'H1T_ele',
        'H1L_ele', 'H1R_ele', 'H2L_ele', 'H2R_ele', 'H2B_ele', 'H2T_ele', 'H3B_ele',
        'H3T_ele', 'H4Y1L_ele', 'H4Y1R_ele', 'H4Y2L_ele', 'H4Y2R_ele', 'H4B_ele',
        'H4T_ele', 'P1Y1_ele', 'P1Y2_ele', 'P1X1_ele', 'P1X2_ele', 'P2X1_ele',
        'P2X2_ele', 'P2Y1_ele', 'P2Y2_ele', 'DP1TL_ele', 'DP1TR_ele', 'DP1BL_ele',
        'DP1BR_ele', 'DP2TL_ele', 'DP2TR_ele', 'DP2BL_ele', 'DP2BR_ele', 'D0U_ele',
        'D0Up_ele', 'D0X_ele', 'D0Xp_ele', 'D0V_ele', 'D0Vp_ele', 'D2U_ele',
        'D2Up_ele', 'D2X_ele', 'D2Xp_ele', 'D2V_ele', 'D2Vp_ele', 'D3pU_ele',
        'D3pUp_ele', 'D3pX_ele', 'D3pXp_ele', 'D3pV_ele', 'D3pVp_ele', 'D3mU_ele',
        'D3mUp_ele', 'D3mX_ele', 'D3mXp_ele', 'D3mV_ele', 'D3mVp_ele', 'D0U_drift',
        'D0Up_drift', 'D0X_drift', 'D0Xp_drift', 'D0V_drift', 'D0Vp_drift',
        'D2U_drift', 'D2Up_drift', 'D2X_drift', 'D2Xp_drift', 'D2V_drift',
        'D2Vp_drift', 'D3pU_drift', 'D3pUp_drift', 'D3pX_drift', 'D3pXp_drift',
        'D3pV_drift', 'D3pVp_drift', 'D3mU_drift', 'D3mUp_drift', 'D3mX_drift',
        'D3mXp_drift', 'D3mV_drift', 'D3mVp_drift']

        

        # Dynamically assign values from targettree to the corresponding variables
    for variables in variables:
        globals()[variables] = TTree[variables].array(library='np')
        #print(variables, globals()[variables].shape)  # Verify shape

        

    

    events = 100 #len(pid)
    detectors_order= [D0U_ele, D0Up_ele,D0X_ele,D0Xp_ele,D0V_ele,D0Vp_ele,
                        D2U_ele,D2Up_ele,D2X_ele,D2Xp_ele,D2V_ele,D2Vp_ele,
                        D3pU_ele,D3pUp_ele,D3pX_ele,D3pXp_ele,D3pV_ele,D3pVp_ele,
                        D3mU_ele,D3mUp_ele,D3mX_ele,D3mXp_ele,D3mV_ele,D3mVp_ele,
                        H1B_ele,H1T_ele,H1L_ele,H1R_ele,H2L_ele,H2R_ele,
                        H2T_ele,H2B_ele,H3B_ele,H3T_ele,H4Y1L_ele,H4Y1R_ele,H4Y2L_ele,
                        H4Y2R_ele,H4B_ele,H4T_ele,P1Y1_ele,P1Y2_ele,P1X1_ele,P1X2_ele,
                        P2X1_ele,P2X2_ele,P2Y1_ele,P2Y2_ele]

    print(f"There are {len(detectors_order)} detectors")

    drift_order =    [D0U_drift, D0Up_drift,D0X_drift,D0Xp_drift,D0V_drift,D0Vp_drift,
                        D2U_drift,D2Up_drift,D2X_drift,D2Xp_drift,D2V_drift,D2Vp_drift,
                        D3pU_drift,D3pUp_drift,D3pX_drift,D3pXp_drift,D3pV_drift,D3pVp_drift,
                        D3mU_drift,D3mUp_drift,D3mX_drift,D3mXp_drift,D3mV_drift,D3mVp_drift]

        
    print("Making hitmatrix")
    hitmatrix = np.zeros((events,49,201))
    #Truth_values = np.zeros((events,2,49,2))
    Truth_values = np.zeros((events,196))
    print(f"Doing {events} events.")
    for event in range(events):
        number_of_tracks = n_tracks.size
        print(f"There are {number_of_tracks} tracks")
        phit = 1
        mhit = 49
        if(n_tracks[event] == 2):
            for track in range(n_tracks[event]):
                nhits = nhits_track[event][track]
                print(f"there are {nhits} hits")
                if( (nhits >= 34) & (nhits <= 48)):
                    if(pid[event][track] > 0):
                        print("Doing the positive track")
                        for i, detector in enumerate(detectors_order, start=1):
                            hit = detector[event][track].astype(int)
                            if(i <= 23):
                                if((hit > 0) & (hit < 1000)):
                                    drift_distance = drift_order[i-1][event][track].astype(float)
                                    hitmatrix[event][i][hit] = drift_distance
                                    Truth_values[event][phit] = hit
                                    Truth_values[event][phit+1] = drift_distance
                                    
                            else:
                                if((hit > 0) & (hit < 1000)):
                                    hitmatrix[event][i][hit] = 1
                                    Truth_values[event][phit] = hit
                                    Truth_values[event][phit+1] = 1
                            phit += 2
                    if(pid[event][track] < 0):
                        print("Doing the negative track")
                        for i, detector in enumerate(detectors_order, start=1):
                            hit = detector[event][track]

                            if(i <= 23):
                                if((hit > 0) & (hit < 1000)):
                                    drift_distance = drift_order[i-1][event][track].astype(float)
                                    hitmatrix[event][i][hit] = drift_distance
                                    Truth_values[event][mhit] = hit
                                    Truth_values[event][mhit+1] = drift_distance
                            else:
                                if((hit > 0) & (hit < 1000)):
                                    hitmatrix[event][i][hit] = 1
                                    Truth_values[event][mhit] = hit
                                    Truth_values[event][mhit+1] = 1
                            mhit += 2

                else:
                    print(f"Event:{event} was not kept due to having {nhits_track[event][track]} hits.")

                        

                
    print("HitMatrix Done")
    return hitmatrix, Truth_values

#Function
# root_file = "rootfiles/DY_Target_500k_080524/merged_trackQA_v2.root"
# hitmatrix, Truth_values = read_root_file(root_file)


