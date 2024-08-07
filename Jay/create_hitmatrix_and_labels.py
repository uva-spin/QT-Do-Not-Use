
import numpy as np
import uproot
from numba import njit, prange


def read_root_file(root_file):
    print("reading root file")
    TTree = uproot.open(root_file+":QA_ana")
 

    #These are the branches of the root TTree that we are interested in. 
    variables = [
        'n_tracks','elementID',"detectorID", 'pid', 'gvx', 'gvy', 'gvz', 'gpx', 'gpy', 'gpz', 'gpt', 'geta', 'gphi',
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

        

    

    #This is the order of the detectors in the root file. DriftChambers, hodoscopes, proptubes.
    detectors_order= [D0U_ele, D0Up_ele,D0X_ele,D0Xp_ele,D0V_ele,D0Vp_ele,
                        D2U_ele,D2Up_ele,D2X_ele,D2Xp_ele,D2V_ele,D2Vp_ele,
                        D3pU_ele,D3pUp_ele,D3pX_ele,D3pXp_ele,D3pV_ele,D3pVp_ele,
                        D3mU_ele,D3mUp_ele,D3mX_ele,D3mXp_ele,D3mV_ele,D3mVp_ele,
                        H1B_ele,H1T_ele,H1L_ele,H1R_ele,H2L_ele,H2R_ele,
                        H2T_ele,H2B_ele,H3B_ele,H3T_ele,H4Y1L_ele,H4Y1R_ele,H4Y2L_ele,
                        H4Y2R_ele,H4B_ele,H4T_ele,P1Y1_ele,P1Y2_ele,P1X1_ele,P1X2_ele,
                        P2X1_ele,P2X2_ele,P2Y1_ele,P2Y2_ele]

    print(f"There are {len(detectors_order)} detectors")

    #Each Driftchamber has a drift distance. 
    drift_order =    [D0U_drift, D0Up_drift,D0X_drift,D0Xp_drift,D0V_drift,D0Vp_drift,
                        D2U_drift,D2Up_drift,D2X_drift,D2Xp_drift,D2V_drift,D2Vp_drift,
                        D3pU_drift,D3pUp_drift,D3pX_drift,D3pXp_drift,D3pV_drift,D3pVp_drift,
                        D3mU_drift,D3mUp_drift,D3mX_drift,D3mXp_drift,D3mV_drift,D3mVp_drift]

    print(f"There are {len(drift_order)} drift distances")
    print("Making hitmatrix")
    events = 1000 #len(pid)
    
    print(f"Doing {events} events.")

    #This loops through several events, filtering good events from bad events.
    #A good event is an event with 2 tracks that goes through each possipble detector.
    #A signal muon can only go through 34 detectors.
    good_events = []
    bad_events = []
    for event in range(events):
        #counter for the hits
   
        nhit_p = 0
        nhit_m = 0
        #Loop over each track in the event
        if(n_tracks[event] == 2):
            for track in range(len(pid[event])):
                #Particle ID is -13 for negative muon and +13 for positive muon.
                particleID = pid[event][track]
                #print(f"The pid is {particleID}.")
                #loops over detectors. 
                for i, detector in enumerate(detectors_order, start=0):
                    #hit is the elementID of that track and detector.
                    hit = detector[event][track]
                    #Filter based on elementID and particle ID. When there is no hit the detector saves as max value postive.
                    if((hit < 1000) & (particleID > 0)):
                        nhit_p +=1
                        #print(hit,i,particleID,nhit_p)
                    if((hit < 1000) & (particleID < 0)):
                        nhit_m +=1
                        #print(hit,i,particleID,nhit_m)


            
            #keeps events with 34 hits in each track
            if((nhit_m == 34) & ((nhit_p == 34))):
                #print("this is a good event!")
                #print(f"Total hits in mup= {nhit_p} and mum has {nhit_m}.")
                good_events = np.append(good_events,event)
            else:
                #print("Bad event!")
                bad_events = np.append(bad_events,event)

    print(f"There are {len(good_events)} good events and {len(bad_events)} bad events!")
    print(good_events)


    #Setup matrices for fast filling. Hitmatrix will be used as the feature in the model. 
    #Truth_values will contain elementID and drift distance. This will be the labels in the model.

    hitmatrix = np.zeros((len(good_events),49,201))
    #Truth_values = np.zeros((events,2,49,2))
    Truth_values = np.zeros((len(good_events),196))
    

    ####################################################
    #hit matrix loop 
    ####################################################
    ith_event = 0
    for event in good_events:
        event = good_events[0]
        event = event.astype(int)
        number_of_tracks = n_tracks[event].size
        #print(f"There are {number_of_tracks} tracks in event: {event}")
        # phit = 0
        # mhit = 48
        index = 0
        
        
        for i, detector in enumerate(detectors_order,start=0):
            for track in range(n_tracks[event]):
                nhits = nhits_track[event][track]
                #print(f"there are {nhits} hits")
                particleID = pid[event][track]
                #print(f"The particle ID is {particleID}")
                if(particleID > 0):
                    #print("Doing the positive track")
                    hit = detector[event][track].astype(int)
                    #Determines if detector is a drift chamber.
                    if(i <= 23):
                        if(hit < 1000):
                            drift_distance = drift_order[i][event][track].astype(float)
                            hitmatrix[ith_event][i][hit] = drift_distance
                            Truth_values[ith_event][index] = hit
                            Truth_values[ith_event][index+1] = drift_distance
                            
                    else:
                        if(hit < 1000):
                            hitmatrix[ith_event][i][hit] = 1
                            Truth_values[ith_event][index] = hit
                            Truth_values[ith_event][index+1] = 1.0
                    index += 2


                if(particleID < 0):
                    #print("Doing the negative track")
                    hit = detector[event][track].astype(int)
                    #Determines if it is in the drift chamber
                    if(i <= 23):
                        if(hit < 1000):
                            drift_distance = drift_order[i][event][track].astype(float)
                            hitmatrix[ith_event][i][hit] = drift_distance
                            Truth_values[ith_event][index] = hit
                            Truth_values[ith_event][index+1] = drift_distance
                    else:
                        if(hit < 1000):
                            hitmatrix[ith_event][i][hit] = 1
                            Truth_values[ith_event][index] = hit
                            Truth_values[ith_event][index+1] = 1.0
                    index += 2

        ith_event += 1

    #End of loop

    elemID = elementID[good_events[0].astype(int)]
    detID = detectorID[good_events[0].astype(int)]

    # print(elemID[elemID < 1000])
    # print(len(elemID[elemID < 1000]))
    # print(detID[detID < 1000])                    

            
    print("HitMatrix Done")
    return hitmatrix, Truth_values, good_events

# #Function
# root_file = "rootfiles/DY_Target_500k_080524/merged_trackQA_v2.root"
# hitmatrix, Truth_values, good_events = read_root_file(root_file)
# print(Truth_values[0].astype(int))
# test = np.reshape(Truth_values,(len(good_events),49,4)).astype(int)
# print(test[0])
# # print(test[:,:,:2])
# # print(test[0][0][:,0])
# # print(test[0][1][:,0])
# # print(len(np.where(test[0][1][:,0] != 0)[0]))
# # print(len(np.where(test[0][0][:,0] != 0)[0]))



