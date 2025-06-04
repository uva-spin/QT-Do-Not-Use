# Data Viewing Tools

This directory contains tools for visualizing and analyzing detector hit data from ROOT files.

## Occupancy.py

`Occupancy.py` is a tool for analyzing and visualizing detector occupancy patterns across multiple ROOT files. It processes detector hit data and generates both individual and combined occupancy plots.

### Features
- Processes multiple ROOT files containing detector hit data
- Generates occupancy plots for individual detector groups
- Creates overall occupancy plots with and without group separators
- Saves accumulated hit data for further analysis
- Supports processing a subset of files for debugging

### Usage
```bash
python Occupancy.py [--directory DIRECTORY]
```

#### Arguments
- `--directory`: Path to the directory containing ROOT files (default: current directory)

### Output
The script generates the following outputs in the `QTracker_Station_Occupancies` directory:
1. Individual detector group plots in separate subdirectories
2. Overall occupancy plots:
   - `overall_occupancy_with_separators.png`
   - `overall_occupancy_no_separators.png`
3. Accumulated hit data:
   - `file_hits_data.json`: Hit counts per file
   - `total_hits_data.json`: Total accumulated hits across all files

### Data Analysis
The saved JSON files can be loaded for further analysis:
```python
# Load total hits
with open('QTracker_Station_Occupancies/total_hits_data.json', 'r') as f:
    total_hits = json.load(f)

# Load per-file hits
with open('QTracker_Station_Occupancies/file_hits_data.json', 'r') as f:
    file_hits = json.load(f)
```

## plot.py

`plot.py` is a tool for creating detailed plots of detector hit patterns. It provides visualization capabilities for analyzing hit distributions and patterns in the detector data.

### Features
- Generates detailed hit pattern visualizations
- Supports custom plot configurations
- Provides options for different visualization styles

### Usage
```bash
python plot.py [options]
```

### Output
The script generates plot files in the specified output directory, showing detailed hit patterns and distributions.

## Requirements
- Python 3.x
- Required Python packages:
  - numpy
  - uproot
  - matplotlib
  - tqdm
  - json

## Installation
```bash
pip install numpy uproot matplotlib tqdm
```

## Notes
- For debugging purposes, you can limit the number of files processed by modifying the `max_files` parameter in `Occupancy.py`
- The scripts expect ROOT files to be in the standard format with detector hit data
- Make sure you have sufficient disk space for the output files, especially when processing large datasets 

## Additional Information

The data that we will be using for these scripts is located in [this Google Drive](https://drive.google.com/drive/folders/1w2mU1nDPTJMfll2MEAYWxTa-N9PvQuhs?usp=share_link). This data is in the SRAW ROOT structure format and is from real experimental data. Please download the entire directory and call the path to that directory when using files such as `Occupancy.py` or `plot.py`. Feel free to edit either of these scripts as needed.