This folder contains the scripts to create and train the QTracker system. To use on rivanna, run the command:

source job_submission.sh

Alternatively, each script in Training_Jobscripts can be run separately using the sbatch command.

If not running on Rivanna, each file in the Python_Files folder can be run independently. Make sure to run the command from this directory, as such:

python Python_Files/script.py [options]
