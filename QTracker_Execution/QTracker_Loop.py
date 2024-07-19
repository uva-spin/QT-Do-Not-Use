###QTracker Rivanna###
# This script is used to reconstruct large amount of data on Rivanna via Slurm job submission.

#####Parent path Directory#####
root_directory = '/project/ptgroup/seaquest/data/digit/02/'

#####Import Functions to Run QTracker#####
from QTracker_Run_Library import *

#####Reconstruction Options#####
dimuon_prob_threshold = 0.75 #Minimum dimuon probability to reconstruct.
timing_cuts = True #Use SRawEvent intime flag for hit filtering

#####Output Options#####
event_prob_output = True #Output the event filter probabilites for reconstructed events
track_quality_output = True #Output the number of drift chamber mismatches for each chamber
target_prob_output = True #Output the probability that the dimuon pair is from the target.
tracks_output = False #Output the element IDs for the identified tracks for all three track finders
metadata_output = True #Output metadata

####Metadata Options#####
#Select which values from the SRawEvent file should be saved to the reconstructed .npy file
runid_output = True #Output the run id
eventid_output = True #Output the event id
spillid_output = True #Output the spill id
triggerbit_output = True #Output the trigger bit for the event
target_pos_output = True #Output the target type (hydrogen, deuterium, etc.)
turnid_output = True #Output the turn id
rfid_output = True #Output the RF ID
intensity_output = True #Output Cherenkov information
trigg_rds_output = True #Output the number of trigger roads activated
occ_output = True #Output the occupancy information
occ_before_cuts = False #If set to true, counts number of hits before timing cuts, if false, outputs occupancies after hit reduction.

import os
import numpy as np
import uproot
import numba
from numba import njit, prange
import tensorflow as tf

root_files = [file for file in os.listdir(root_directory) if file.endswith('.root')]
    
for i, root_file in enumerate(root_files):
    process_file(root_file, root_directory, i, max_ele, dimuon_prob_threshold, means, stds, kin_means, kin_stds)