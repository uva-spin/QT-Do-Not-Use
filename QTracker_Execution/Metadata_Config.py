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
#Only affects output if using .root file.
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