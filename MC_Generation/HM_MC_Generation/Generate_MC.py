import numpy as np
import uproot
import numba
from numba import njit, prange
import random
import gc
import os
import sys

#Set the path to the NIM3 files here. A path that works in Rivanna is included here.
nim3_path = "/project/ptgroup/QTracker_Training/NIM3/"

# Check if the script is run without a ROOT file or with the script name as input.
if len(sys.argv) != 2:
    print("Usage: python script_name.py <input_file.root>")
    quit()

root_file = sys.argv[1]  # Takes the first command-line argument as the input file path.

# Check if the input file has a valid extension
valid_extensions = ('.root')
file_extension = os.path.splitext(root_file)[1]
if file_extension not in valid_extensions:
    print("Invalid input file format. Supported formats: .root")
    quit()

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
def hit_matrix_partial_track(detectorid,elementid,drifttime,hits,drift,station): #Convert into hit matrices
    for j in prange (len(detectorid)):
        #St 1
        if(station==1):
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
        elif(station==3):
            if (detectorid[j]>18 and (detectorid[j]<31)) or ((detectorid[j]>38) and (detectorid[j]<47)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
        #St 4
        elif(station==4):
            if ((detectorid[j]>40) and (detectorid[j]<55)):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
    return hits,drift

@numba.jit(nopython=True)
def hit_matrix_mc(detectorid,elementid,drifttime,hits,drift): #Convert into hit matrices
    for j in prange (len(detectorid)):
        if ((detectorid[j]<7) or ((detectorid[j]>12) and (detectorid[j]<55))) and ((random.random()<0.94) or (detectorid[j]>30)):
            hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
            drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
    return hits,drift

@njit(parallel=True)
def evaluate_finder(testin, testdrift, predictions):
    reco_in = np.zeros((len(testin), 68, 2))
    
    def process_entry(i, dummy, j_offset):
        j = dummy if dummy <= 5 else dummy + 6
        if dummy > 11:
            if predictions[i][12] > 0:
                j = dummy + 6
            elif predictions[i][12] < 0:
                j = dummy + 12

        if dummy > 17:
            j = 2 * (dummy - 18) + 30 if predictions[i][2 * (dummy - 18) + 30] > 0 else 2 * (dummy - 18) + 31

        if dummy > 25:
            j = dummy + 20

        k = abs(predictions[i][dummy + j_offset])
        sign = k / predictions[i][dummy + j_offset] if k > 0 else 0
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

    for i in prange(predictions.shape[0]):
        for dummy in prange(34):
            process_entry(i, dummy, 0)
        
        for dummy in prange(34):
            process_entry(i, dummy, 34)

    return reco_in

max_ele = [200, 200, 168, 168, 200, 200, 128, 128,  112,  112, 128, 128, 134, 134, 
           112, 112, 134, 134,  20,  20,  16,  16,  16,  16,  16,  16,
        72,  72,  72,  72,  72,  72,  72,  72, 200, 200, 168, 168, 200, 200,
        128, 128,  112,  112, 128, 128, 134, 134, 112, 112, 134, 134,
        20,  20,  16,  16,  16,  16,  16,  16,  72,  72,  72,  72,  72,
        72,  72,  72]

def generate_e906():

    #Load MC Data
    targettree = uproot.open(root_file+':QA_ana')
    detectorid=targettree["detectorID"].arrays(library="np")["detectorID"]
    elementid=targettree["elementID"].arrays(library="np")["elementID"]
    driftdistance=targettree["driftDistance"].arrays(library="np")["driftDistance"]
    pid=targettree['pid'].arrays(library="np")['pid']
    px=targettree["gpx"].arrays(library="np")['gpx']
    py=targettree["gpy"].arrays(library="np")['gpy']
    pz=targettree["gpz"].arrays(library="np")['gpz']
    vx=targettree["gvx"].arrays(library="np")['gvx']
    vy=targettree["gvy"].arrays(library="np")['gvy']
    vz=targettree["gvz"].arrays(library="np")['gvz']
    nhits=targettree['nhits'].array(library='np')
    n_events = len(pid)
    
    #Initialize hit matrix, drift matrix, and kinematics storage arrays.
    hits = np.zeros((n_events,54,201))
    drift = np.zeros((n_events,54,201))
    kinematics = np.zeros((n_events,9))
    #Place tracks on the hit matrix.
    for n in range (n_events):
        hits[n],drift[n]=hit_matrix_mc(detectorid[n],elementid[n],driftdistance[n],hits[n],drift[n])
        if(pid[n][0]>0):
            kinematics[n][0]=px[n][0]
            kinematics[n][1]=py[n][0]
            kinematics[n][2]=pz[n][0]
            kinematics[n][3]=px[n][1]
            kinematics[n][4]=py[n][1]
            kinematics[n][5]=pz[n][1]
        if(pid[n][0]<0):
            kinematics[n][0]=px[n][1]
            kinematics[n][1]=py[n][1]
            kinematics[n][2]=pz[n][1]
            kinematics[n][3]=px[n][0]
            kinematics[n][4]=py[n][0]
            kinematics[n][5]=pz[n][0]
        kinematics[n][6]=vx[n][0]
        kinematics[n][7]=vy[n][0]
        kinematics[n][8]=vz[n][0]
    del detectorid, elementid,driftdistance


    #Import NIM3 events and put them on the hit matrices.
    #Randomly choose between 1 and 6 events per station.
    filelist=['output_part1.root:tree_nim3','output_part2.root:tree_nim3','output_part3.root:tree_nim3',
             'output_part4.root:tree_nim3','output_part5.root:tree_nim3','output_part6.root:tree_nim3',
             'output_part7.root:tree_nim3','output_part8.root:tree_nim3','output_part9.root:tree_nim3']
    targettree = uproot.open(nim3_path+random.choice(filelist))
    detectorid_nim3=targettree["det_id"].arrays(library="np")["det_id"]
    elementid_nim3=targettree["ele_id"].arrays(library="np")["ele_id"]
    driftdistance_nim3=targettree["drift_dist"].arrays(library="np")["drift_dist"]
    
    for n in range (n_events): #Create NIM3 events
        g=random.choice([1,2,3,4,5,6])
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
      

    
    return hits.astype(bool), drift, kinematics

print('Generating hit matrices...')
hits, drift, truth = generate_e906()
print('Saving...')

file_name = os.path.splitext(root_file)[0]

# Save the outputs to a NumPy file
np.savez_compressed(file_name+"_events.npz", hits=hits, drift=drift, truth=truth)
print('File saved at '+file_name+"_events.npz"

