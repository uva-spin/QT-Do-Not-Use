This folder contains the code to split generated MC data into training and validation data sets, as well as to resample it to have a flat mass distribution over the expected mass range for E906 and E1039.

To use it, place the generated MC data in the folder MC/, following the naming conventions described in that file.

Then, simply execute 

python Data_Prep.py

NOTE: This code requires a valid installation of PyRoot, which Rivanna does not have. This code must be done using a machine with PyRoot installed.
