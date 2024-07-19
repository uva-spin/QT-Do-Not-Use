import os
import numpy as np
import uproot
import numba
from numba import njit, prange
import tensorflow as tf

network_path = '/scratch/acc5dn/QTracker_Run_Refactor/Networks/'

def save_explanation():
    explanation = []
    n_columns = 0
    if event_prob_output: 
        explanation.append(f"Event Filter Probabilites: Columns {n_columns}:{n_columns+6}")
        n_columns+=2
    explanation.append(f"Muon Z-Vertex: Columns {n_columns}:{n_columns+1}")
    n_columns+=2
    explanation.append(f"All Vertex Kinematic Predictions: Columns {n_columns}:{n_columns+6}")
    n_columns+=6
    explanation.append(f"All Vertex Vertex Predictions: Columns {n_columns}:{n_columns+3}")
    n_columns+=3
    explanation.append(f"Z Vertex Kinematic Predictions: Columns {n_columns}:{n_columns+6}")
    n_columns+=6
    explanation.append(f"Z Vertex Vertex Predictions: Columns {n_columns}:{n_columns+3}")
    n_columns+=3
    explanation.append(f"Target Vertex Kinematic Predictions: Columns {n_columns}:{n_columns+6}")
    n_columns+=6
    explanation.append(f"Dump Vertex Kinematic Predictions: Columns {n_columns}:{n_columns+6}")
    n_columns+=6
    if target_prob_output:
        explanation.append(f"Target Probability: Column {n_columns}")
        n_columns+=1
    if track_quality_output:
        explanation.append(f"Muon Track Quality: Colums {n_columns}:{n_columns+12}")
        n_columns+=12
        explanation.append(f"Dimuon Track Quality: Colums {n_columns}:{n_columns+12}")
        n_columns+=12
    if tracks_output: 
        explanation.append(f"Pos Muon Track: {n_columns}:{n_columns+34}")
        n_columns+=34
        explanation.append(f"Neg Muon Track: {n_columns}:{n_columns+34}")
        n_columns+=34
        explanation.append(f"All Vertex Track: {n_columns}:{n_columns+68}")
        n_columns+=68
        explanation.append(f"Z Vertex Track: {n_columns}:{n_columns+68}")
        n_columns+=68
        explanation.append(f"Target Vertex Track: {n_columns}:{n_columns+68}")
        n_columns+=68
        explanation.append(f"Dump Vertex Track: {n_columns}:{n_columns+68}")
        n_columns+=68
    if metadata_output:
        if runid_output:
            explanation.append(f"Run ID: Column {n_columns}")
            n_columns+=1
        if eventid_output:
            explanation.append(f"Event ID: Column {n_columns}")
            n_columns+=1
        if spillid_output:
            explanation.append(f"Spill ID: Column {n_columns}")
            n_columns+=1
        if triggerbit_output:
            explanation.append(f"Trigger Bits: Column {n_columns}")
            n_columns+=1
        if target_pos_output:
            explanation.append(f"Target Positions: Column {n_columns}")
            n_columns+=1
        if turnid_output:
            explanation.append(f"Turn ID: Column {n_columns}")
            n_columns+=1
        if rfid_output:
            explanation.append(f"RFID: Column {n_columns}")
            n_columns+=1
        if intensity_output:
            explanation.append(f"Cherenkov Information: Columns {n_columns}:{n_columns+32}")
            n_columns+=32
        if trigg_rds_output:
            explanation.append(f"Number of Trigger Roads: Column {n_columns}")
            n_columns+=1
        if occ_output:
            if occ_before_cuts:explanation.append(f"Detector Occupancies before cuts: Columns {n_columns}:{n_columns+54}")
            else:explanation.append(f"Detector Occupancies after cuts: Columns {n_columns}:{n_columns+54}")
            n_columns+=54

    filename= f'reconstructed_columns.txt'
    with open(filename,'w') as file:
        file.write('Explanation of Columns:\n\n')
        for info in explanation:
            file.write(f"{info}\n")    

save_explanation()            

def save_output():
    # After processing through all models, the results are aggregated based on options at top,
    # and the final dataset is prepared.

    # The reconstructed kinematics and vertex information are normalized
    # using predefined means and standard deviations before saving.
    row_count = 0
    definitions_string = []	
    if file_extension == '.root':
        metadata = []
        metadata_row_count = 0
        metadata_string = []
        if runid_output:metadata.append(runid)
        if eventid_output:metadata.append(eventid)
        if spillid_output:metadata.append(spill_id)
        if triggerbit_output:metadata.append(trigger_bit)
        if target_pos_output:metadata.append(target_position)
        if turnid_output:metadata.append(turnid)
        if rfid_output:metadata.append(rfid)
        if intensity_output:metadata.append(intensity)
        if trigg_rds_output:metadata.append(n_roads)
        if occ_output:
            if occ_before_cuts:metadata.append(n_hits)
            else:metadata.append(np.sum(hits,axis=2))#Calculates the occupanceis from the Hit Matrix 
        metadata = np.column_stack(metadata)
    if file_extension == '.npz':
        metadata = truth
    output = []
    if event_prob_output: output.append(event_classification_probabilies)
        output.append(all_predictions)
    if target_prob_output:
        output.append(target_dump_prob[:,1])
    if track_quality_output:
        output.append(muon_track_quality)
        output.append(dimuon_track_quality)
    if tracks_output: output.append(tracks)
    if metadata_output: output.append(metadata)
    output_data = np.column_stack(output)

    base_filename = 'Reconstructed/' + os.path.basename(root_file).split('.')[0]
    os.makedirs("Reconstructed", exist_ok=True)  # Ensure the output directory exists.
    np.save(base_filename + '_reconstructed.npy', output_data)  # Save the final dataset.
    print(f"File {base_filename}_reconstructed.npy has been saved successfully.")

        
kin_means = np.array([ 2.00, 0.00, 35.0, -2.00, -0.00, 35.0 ])
kin_stds = np.array([ 0.6, 1.2, 10.00, 0.60, 1.20, 10.00 ])

vertex_means=np.array([0,0,-300])
vertex_stds=np.array([10,10,300])

means = np.concatenate((kin_means,vertex_means))
stds = np.concatenate((kin_stds,vertex_stds))

# Function to convert raw detector data into a structured hit matrix.
@njit()
def hit_matrix(detectorid,elementid,drifttime,tdctime,intime,hits,drift,tdc): #Convert into hit matrices
    if(timing_cuts==False):intime[:]=2
    for j in prange (len(detectorid)):
        if ((detectorid[j]<7) or (detectorid[j]>12)) and (intime[j]>0):
            if (tdc[int(detectorid[j])-1][int(elementid[j]-1)]==0) or (tdctime[j]<tdc[int(detectorid[j])-1][int(elementid[j]-1)]):
                hits[int(detectorid[j])-1][int(elementid[j]-1)]=1
                drift[int(detectorid[j])-1][int(elementid[j]-1)]=drifttime[j]
                tdc[int(detectorid[j])-1][int(elementid[j]-1)]=tdctime[j]
    return hits,drift,tdc

max_ele = [200, 200, 168, 168, 200, 200, 128, 128,  112,  112, 128, 128, 134, 134, 
           112, 112, 134, 134,  20,  20,  16,  16,  16,  16,  16,  16,
        72,  72,  72,  72,  72,  72,  72,  72, 200, 200, 168, 168, 200, 200,
        128, 128,  112,  112, 128, 128, 134, 134, 112, 112, 134, 134,
        20,  20,  16,  16,  16,  16,  16,  16,  72,  72,  72,  72,  72,
        72,  72,  72]

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

# Function to remove closely spaced hits that are likely not real particle interactions (cluster hits).
@njit(parallel=True, forceobj=True)
def declusterize(hits, drift, tdc):
    # This function iterates over hits and removes clusters of hits that are too close
    # together, likely caused by noise or multiple hits from a single particle passing
    # through the detector. It's an important step in cleaning the data for analysis.
    for k in prange(len(hits)):
        for i in range(54):
            if(i<30 or i>45):
                for j in range(100):#Work from both sides
                    if(hits[k][i][j]==1 and hits[k][i][j+1]==1):
                        if(hits[k][i][j+2]==0):#Two hits
                            if(drift[k][i][j]>0.4 and drift[k][i][j+1]>0.9):#Edge hit check
                                hits[k][i][j+1]=0
                                drift[k][i][j+1]=0
                                tdc[k][i][j+1]=0
                            elif(drift[k][i][j+1]>0.4 and drift[k][i][j]>0.9):#Edge hit check
                                hits[k][i][j]=0
                                drift[k][i][j]=0
                                tdc[k][i][j]=0
                            if(abs(tdc[k][i][j]-tdc[k][i][j+1])<8):#Electronic Noise Check
                                hits[k][i][j+1]=0
                                drift[k][i][j+1]=0
                                tdc[k][i][j+1]=0
                                hits[k][i][j]=0
                                drift[k][i][j]=0
                                tdc[k][i][j]=0

                        else:#Check larger clusters for Electronic Noise
                            n=2
                            while(hits[k][i][j+n]==1):n=n+1
                            dt_mean = 0
                            for m in range(n-1):
                                dt_mean += (tdc[k][i][j+m]-tdc[k][i][j+m+1])
                            dt_mean = dt_mean/(n-1)
                            if(dt_mean<10):
                                for m in range(n):
                                    hits[k][i][j+m]=0
                                    drift[k][i][j+m]=0
                                    tdc[k][i][j+m]=0
                    if(hits[k][i][200-j]==1 and hits[k][i][199-j]):
                        if(hits[k][i][198-j]==0):
                            if(drift[k][i][200-j]>0.4 and drift[k][i][199-j]>0.9):  # Edge hit check
                                hits[k][i][199-j]=0
                                drift[k][i][199-j]=0
                            elif(drift[k][i][199-j]>0.4 and drift[k][i][200-j]>0.9):  # Edge hit check
                                hits[k][i][200-j]=0
                                drift[k][i][200-j]=0
                            if(abs(tdc[k][i][200-j]-tdc[k][i][199-j])<8):  # Electronic Noise Check
                                hits[k][i][199-j]=0
                                drift[k][i][199-j]=0
                                tdc[k][i][199-j]=0
                                hits[k][i][200-j]=0
                                drift[k][i][200-j]=0
                                tdc[k][i][200-j]=0
                        else:  # Check larger clusters for Electronic Noise
                            n=2
                            while(hits[k][i][200-j-n]==1): n=n+1
                            dt_mean = 0
                            for m in range(n-1):
                                dt_mean += abs(tdc[k][i][200-j-m]-tdc[k][i][200-j-m-1])
                            dt_mean = dt_mean/(n-1)
                            if(dt_mean<10):
                                for m in range(n):
                                    hits[k][i][200-j-m]=0
                                    drift[k][i][200-j-m]=0
                                    tdc[k][i][200-j-m]=0                           


# Specify the directory containing the root files
i = 0

# Drift chamber mismatch calculation
def calc_mismatches(track):
    results = []
    for pos_slice, neg_slice in [(slice(0, 6), slice(34, 40)), (slice(6, 12), slice(40, 46)), (slice(12, 18), slice(46, 52))]:
        # Compare even indices with odd indices for the 0th component of the final dimension
        even_pos_indices = track[:, pos_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 0]
        odd_pos_indices = track[:, pos_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 1]
        even_neg_indices = track[:, neg_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 0]
        odd_neg_indices = track[:, neg_slice, 0].reshape(track.shape[0], -1, 2)[:, :, 1]

        results.extend([
            np.sum(abs(even_pos_indices - odd_pos_indices) > 1, axis=1),
            np.sum(abs(even_neg_indices - odd_neg_indices) > 1, axis=1),
            np.sum(track[:, pos_slice, 2] == 0, axis=1),
            np.sum(track[:, neg_slice, 2] == 0, axis=1)
        ])
    
    return np.array(results)

def load_model(network):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    return tf.keras.models.load_model(network_path+network)

def process_file(file_path, file_extension, max_ele, dimuon_prob_threshold, means, stds, kin_means, kin_stds):
    try:
        if file_extension == '.root':
            # Read in data from the ROOT file.
            targettree = uproot.open(file_path + ":save")
            detectorid = targettree["fAllHits.detectorID"].arrays(library="np")["fAllHits.detectorID"]
            elementid = targettree["fAllHits.elementID"].arrays(library="np")["fAllHits.elementID"]
            driftdistance = targettree["fAllHits.driftDistance"].arrays(library="np")["fAllHits.driftDistance"]
            tdctime = targettree["fAllHits.tdcTime"].arrays(library="np")["fAllHits.tdcTime"]
            intime = targettree["fAllHits.flag"].arrays(library="np")["fAllHits.flag"]

            hits = np.zeros((len(detectorid), 54, 201), dtype=bool)
            drift = np.zeros((len(detectorid), 54, 201))
            tdc = np.zeros((len(detectorid), 54, 201), dtype=int)

            for n in range(len(detectorid)):
                hits[n], drift[n], tdc[n] = hit_matrix(detectorid[n], elementid[n], driftdistance[n], tdctime[n], intime[n], hits[n], drift[n], tdc[n])

            declusterize(hits, drift, tdc)

        elif file_extension == '.npz':
            generated = np.load(file_path)
            hits = generated["hits"]
            drift = generated["drift"]
            truth = generated["truth"]

        print("Loaded events")

        model = load_model('Networks/event_filter')
        probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
        event_classification_probabilities = probability_model.predict(hits, batch_size=256, verbose=0)

        filt = event_classification_probabilities[:, 1] > dimuon_prob_threshold
        hits, drift = hits[filt], drift[filt]

        pos_model = load_model('Track_Finder_Pos')
        pos_predictions = pos_model.predict(hits, verbose=0)
        neg_model = load_model('Track_Finder_Neg')
        neg_predictions = neg_model.predict(hits, verbose=0)
        predictions = (np.round(np.column_stack((pos_predictions, neg_predictions)) * max_ele)).astype(int)

        muon_track = evaluate_finder(hits, drift, predictions)

        pos_recon_model = load_model('Reconstruction_Pos')
        pos_pred = pos_recon_model.predict(muon_track[:, :34, :2], verbose=0)
        neg_recon_model = load_model('Reconstruction_Neg')
        neg_pred = neg_recon_model.predict(muon_track[:, 34:, :2], verbose=0)

        muon_track_quality = calc_mismatches(muon_track).T
        filt1 = ((muon_track_quality[0::4] < 2) & (muon_track_quality[1::4] < 2) & (muon_track_quality[2::4] < 3) & (muon_track_quality[3::4] < 3)).all(axis=0)

        hits, drift = hits[filt1], drift[filt1]

        if file_extension == '.root':
            runid = targettree["fRunID"].arrays(library="np")["fRunID"][filt][filt1]
            eventid = targettree["fEventID"].arrays(library="np")["fEventID"][filt][filt1]
            spill_id = targettree["fSpillID"].arrays(library="np")["fSpillID"][filt][filt1]
            trigger_bit = targettree["fTriggerBits"].arrays(library="np")["fTriggerBits"][filt][filt1]
            target_position = targettree["fTargetPos"].arrays(library="np")["fTargetPos"][filt][filt1]
            turnid = targettree["fTurnID"].arrays(library="np")["fTurnID"][filt][filt1]
            rfid = targettree["fRFID"].arrays(library="np")["fRFID"][filt][filt1]
            intensity = targettree["fIntensity[33]"].arrays(library="np")["fIntensity[33]"][filt][filt1]
            n_roads = targettree["fNRoads[4]"].arrays(library="np")["fNRoads[4]"][filt][filt1]
            n_hits = targettree["fNHits[55]"].arrays(library="np")["fNHits[55]"][filt][filt1]

        elif file_extension == '.npz':
            truth = truth[filt][filt1]

        print("Filtered Events")
        if len(hits) > 0:
            track_finder_all_model = load_model('Track_Finder_All')
            predictions = (np.round(track_finder_all_model.predict(hits, verbose=0) * max_ele)).astype(int)
            all_vtx_track = evaluate_finder(hits, drift, predictions)[:, :, :2]

            reco_all_model = load_model('Reconstruction_All')
            reco_kinematics = reco_all_model.predict(all_vtx_track, batch_size=8192, verbose=0)

            vertex_input = np.concatenate((reco_kinematics.reshape((len(reco_kinematics), 3, 2)), all_vtx_track), axis=1)

            vertexing_all_model = load_model('Vertexing_All')
            reco_vertex = vertexing_all_model.predict(vertex_input, batch_size=8192, verbose=0)

            all_vtx_reco = np.concatenate((reco_kinematics, reco_vertex), axis=1)

            track_finder_z_model = load_model('Track_Finder_Z')
            predictions = (np.round(track_finder_z_model.predict(hits, verbose=0) * max_ele)).astype(int)
            z_vtx_track = evaluate_finder(hits, drift, predictions)[:, :, :2]

            reco_z_model = load_model('Reconstruction_Z')
            reco_kinematics = reco_z_model.predict(z_vtx_track, batch_size=8192, verbose=0)

            vertex_input = np.concatenate((reco_kinematics.reshape((len(reco_kinematics), 3, 2)), z_vtx_track), axis=1)

            vertexing_z_model = load_model('Vertexing_Z')
            reco_vertex = vertexing_z_model.predict(vertex_input, batch_size=8192, verbose=0)

            z_vtx_reco = np.concatenate((reco_kinematics, reco_vertex), axis=1)

            track_finder_target_model = load_model('Track_Finder_Target')
            predictions = (np.round(track_finder_target_model.predict(hits, verbose=0) * max_ele)).astype(int)
            target_track = evaluate_finder(hits, drift, predictions)

            reco_target_model = load_model('Reconstruction_Target')
            target_vtx_reco = reco_target_model.predict(target_track[:, :, :2], batch_size=8192, verbose=0)

            track_finder_dump_model = load_model('Track_Finder_Dump')
            predictions = (np.round(track_finder_dump_model.predict(hits, verbose=0) * max_ele)).astype(int)
            dump_track = evaluate_finder(hits, drift, predictions)[:, :, :2]

            reco_dump_model = load_model('Reconstruction_Dump')
            dump_vtx_reco = reco_dump_model.predict(dump_track, batch_size=8192, verbose=0)

            dimuon_track_quality = calc_mismatches(target_track).T

            reco_kinematics = np.concatenate((event_classification_probabilities[:, 1], pos_pred, neg_pred, all_vtx_reco, z_vtx_reco, target_vtx_reco, dump_vtx_reco, muon_track_quality, dimuon_track_quality), axis=1)

            tracks = np.column_stack((muon_track[:, :, :2], all_vtx_track, z_vtx_track, target_track[:, :, :2], dump_track))

            target_dump_input = np.column_stack((reco_kinematics, tracks.reshape((len(tracks), (68 * 2 * 5)))))

            target_dump_filter_model = load_model('target_dump_filter')
            target_dump_pred = target_dump_filter_model.predict(target_dump_input, batch_size=512, verbose=0)
            target_dump_prob = np.exp(target_dump_pred) / np.sum(np.exp(target_dump_pred), axis=1, keepdims=True)

            all_predictions = np.column_stack((pos_pred, neg_pred, all_vtx_reco * stds + means, z_vtx_reco * stds + means, target_vtx_reco * kin_stds + kin_means, dump_vtx_reco * kin_stds + kin_means))

            print("Found", len(all_predictions), "Dimuons in file.")
            
            save_output()
        else:
            print("No events meeting dimuon criteria.")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")