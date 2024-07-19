#!/bin/bash

# Submit the Event Filter Training Job
job1_id=$(sbatch Training_Jobscripts/jobscript_Event_Filter.sh | awk '{print $4}')

# Submit the Muon Track Finding Jobs, start once the event filter is done training.
job2_id=$(sbatch --dependency=afterok:$job1_id Training_Jobscripts/jobscript_TFP.sh | awk '{print $4}')
job3_id=$(sbatch --dependency=afterok:$job1_id Training_Jobscripts/jobscript_TFN.sh | awk '{print $4}')

# Once Muon Finders are trained, start training muon vertex reconstruction and dimuon track finders.
job4_id=$(sbatch --dependency=afterok:$job2_id:$job3_id Training_Jobscripts/jobscript_TFA.sh | awk '{print $4}')
job5_id=$(sbatch --dependency=afterok:$job2_id:$job3_id Training_Jobscripts/jobscript_TFZ.sh | awk '{print $4}')
job6_id=$(sbatch --dependency=afterok:$job2_id:$job3_id Training_Jobscripts/jobscript_TFT.sh | awk '{print $4}')
job7_id=$(sbatch --dependency=afterok:$job2_id:$job3_id Training_Jobscripts/jobscript_TFD.sh | awk '{print $4}')
job8_id=$(sbatch --dependency=afterok:$job2_id:$job3_id Training_Jobscripts/jobscript_Muon_Vertexing.sh | awk '{print $4}')

# Submit the Reconstruction trainings after their respective track finders are done training.
job9_id=$(sbatch --dependency=afterok:$job4_id Training_Jobscripts/jobscript_Kin_All.sh | awk '{print $4}')
job10_id=$(sbatch --dependency=afterok:$job5_id Training_Jobscripts/jobscript_Kin_Z.sh | awk '{print $4}')
job11_id=$(sbatch --dependency=afterok:$job6_id Training_Jobscripts/jobscript_Kin_Target.sh | awk '{print $4}')
job12_id=$(sbatch --dependency=afterok:$job7_id Training_Jobscripts/jobscript_Kin_Dump.sh | awk '{print $4}')

# Submit the target-dump filter training once everything else is done.
job13_id=$(sbatch --dependency=afterok:$job8_id:$job9_id:$job10_id:$job11_id:$job12_id Training_Jobscripts/jobscript_Target_Reco_Generation.sh | awk '{print $4}')
job14_id=$(sbatch --dependency=afterok:$job8_id:$job9_id:$job10_id:$job11_id:$job12_id Training_Jobscripts/jobscript_Dump_Reco_Generation.sh | awk '{print $4}')
sbatch --dependency=afterok:$job13_id:$job14_id Training_Jobscripts/jobscript_Target_Dump_Filter.sh
