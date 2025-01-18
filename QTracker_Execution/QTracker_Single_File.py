###QTracker Execution###
#This script is used for reconstructing a single root or numpy file.
#Usage:
"""""""""""""""
python QTracker_Run.py /path/to/file.root|.npz
"""""""""""""""

#####Import Functions to Run QTracker#####
from QTracker_Run_Library_Test import *
from Metadata_Config import *

import os
import numpy as np
import uproot  # For reading ROOT files, a common data format in particle physics.
import numba  # Just-In-Time (JIT) compiler for speeding up Python code.
from numba import njit, prange  # njit for compiling functions, prange for parallel loops.
import tensorflow as tf  # For using machine learning models.
import sys

# Check if the script is run without a ROOT file or with the script name as input.
if len(sys.argv) != 2:
    print("Usage: python script_name.py <input_file.root|.npz>")
    quit()

root_file = sys.argv[1]  # Takes the first command-line argument as the input file path.

# Check if the input file has a valid extension
valid_extensions = ('.root', '.npz')
file_extension = os.path.splitext(root_file)[1]
if file_extension not in valid_extensions:
    print("Invalid input file format. Supported formats: .root, .npy")
    quit()