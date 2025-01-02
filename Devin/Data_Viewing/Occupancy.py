import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def process_station_occupancy(root_dir, detector_groups):
    station_data = {}
    valid_detector_ids = {det['id'] for group in detector_groups for det in group['detectors']}
    
    for group in detector_groups:
        station_data[group['label']] = {det['id']: np.zeros(det['elements']) for det in group['detectors']}

    for filename in os.listdir(root_dir):
        if not filename.endswith(".root"):
            continue
            
        file_path = os.path.join(root_dir, filename)
        print(f"Processing {filename}")
        
        try:
            with uproot.open(file_path + ":save") as file:
                detector_ids = np.concatenate(file["fAllHits.detectorID"].array(library="np"))
                element_ids = np.concatenate(file["fAllHits.elementID"].array(library="np"))
                
                # Check for invalid detector IDs
                invalid_ids = set(np.unique(detector_ids)) - valid_detector_ids
                if invalid_ids:
                    print(f"Warning: Found invalid detector IDs: {invalid_ids}")
                    
                for group in detector_groups:
                    for detector in group['detectors']:
                        mask = (detector_ids == detector['id'])
                        if np.any(mask):
                            valid_elements = element_ids[mask]
                            valid_elements = valid_elements[valid_elements < detector['elements']]
                            if len(valid_elements) > 0:
                                hist, _ = np.histogram(
                                    valid_elements,
                                    bins=detector['elements'],
                                    range=(0, detector['elements'])
                                )
                                station_data[group['label']][detector['id']] += hist
                            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    return station_data

def plot_station_heatmaps(station_data, detector_groups, output_dir):
    for group in detector_groups:
        # Create data matrix with proper padding
        max_elements = max(det['elements'] for det in group['detectors'])
        data_matrix = np.zeros((len(group['detectors']), max_elements))
        
        for idx, detector in enumerate(group['detectors']):
            data = station_data[group['label']][detector['id']]
            data_matrix[idx, :len(data)] = data

        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = LinearSegmentedColormap.from_list('yellow_colormap', ['#000000', '#FFFF00'], N=256)
        
        im = ax.imshow(data_matrix.T, aspect='auto', origin='lower', 
                      interpolation='nearest', cmap=cmap)
        
        plt.colorbar(im, label='Hit Counts')
        
        ax.set_xlabel('Detector')
        ax.set_ylabel('Element ID')
        ax.set_title(f'{group["label"]} Occupancy')
        
        ax.set_xticks(range(len(group['detectors'])))
        ax.set_xticklabels([det['name'] for det in group['detectors']], 
                          rotation=45, ha='right')
        
        plt.tight_layout()
        output_file = os.path.join(output_dir, 
                                 f'occupancy_{group["label"].replace(" ", "_").replace("+", "p").replace("-", "m")}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    detector_groups = [
    {'label': 'Station 1', 'detectors': [
        {'name': 'D0V', 'elements': 201, 'id': 5},
        {'name': 'D0Vp', 'elements': 201, 'id': 6},
        {'name': 'D0Xp', 'elements': 160, 'id': 4},
        {'name': 'D0X', 'elements': 160, 'id': 3},
        {'name': 'D0U', 'elements': 201, 'id': 1},
        {'name': 'D0Up', 'elements': 201, 'id': 2}
    ]},
    {'label': 'Hodo', 'detectors': [
        {'name': 'H1L', 'elements': 20, 'id': 33},
        {'name': 'H1R', 'elements': 20, 'id': 34},
        {'name': 'H1B', 'elements': 23, 'id': 31},
        {'name': 'H1T', 'elements': 23, 'id': 32}
    ]},
    {'label': 'DP-1', 'detectors': [
        {'name': 'DP1TL', 'elements': 80, 'id': 55},
        {'name': 'DP1TR', 'elements': 80, 'id': 56},
        {'name': 'DP1BL', 'elements': 80, 'id': 57},
        {'name': 'DP1BR', 'elements': 80, 'id': 58}
    ]},
    {'label': 'Station 2', 'detectors': [
        {'name': 'D2V', 'elements': 128, 'id': 13},
        {'name': 'D2Vp', 'elements': 128, 'id': 14},
        {'name': 'D2Xp', 'elements': 112, 'id': 15},
        {'name': 'D2X', 'elements': 112, 'id': 16},
        {'name': 'D2U', 'elements': 128, 'id': 17},
        {'name': 'D2Up', 'elements': 128, 'id': 18}
    ]},
    {'label': 'Hodo', 'detectors': [
        {'name': 'H2R', 'elements': 19, 'id': 36},
        {'name': 'H2L', 'elements': 19, 'id': 35},
        {'name': 'H2T', 'elements': 16, 'id': 38},
        {'name': 'H2B', 'elements': 16, 'id': 37}
    ]},
    {'label': 'DP-2', 'detectors': [
        {'name': 'DP2TL', 'elements': 48, 'id': 59},
        {'name': 'DP2TR', 'elements': 48, 'id': 60},
        {'name': 'DP2BL', 'elements': 48, 'id': 61},
        {'name': 'DP2BR', 'elements': 48, 'id': 62}
    ]},
    {'label': 'Station 3+', 'detectors': [
        {'name': 'D3pVp', 'elements': 134, 'id': 19},
        {'name': 'D3pV', 'elements': 134, 'id': 20},
        {'name': 'D3pXp', 'elements': 116, 'id': 21},
        {'name': 'D3pX', 'elements': 116, 'id': 22},
        {'name': 'D3pUp', 'elements': 134, 'id': 23},
        {'name': 'D3pU', 'elements': 134, 'id': 24}
    ]},
    {'label': 'Station 3-', 'detectors': [
        {'name': 'D3mVp', 'elements': 134, 'id': 25},
        {'name': 'D3mV', 'elements': 134, 'id': 26},
        {'name': 'D3mXp', 'elements': 116, 'id': 27},
        {'name': 'D3mX', 'elements': 116, 'id': 28},
        {'name': 'D3mUp', 'elements': 134, 'id': 29},
        {'name': 'D3mU', 'elements': 134, 'id': 30}
    ]},
    {'label': 'Hodo', 'detectors': [
        {'name': 'H3T', 'elements': 16, 'id': 40},
        {'name': 'H3B', 'elements': 16, 'id': 39}
    ]},
    {'label': 'Prop', 'detectors': [
        {'name': 'P1Y1', 'elements': 72, 'id': 47},
        {'name': 'P1Y2', 'elements': 72, 'id': 48}
    ]},
    {'label': 'Hodo', 'detectors': [
        {'name': 'H4Y1R', 'elements': 16, 'id': 42},
        {'name': 'H4Y1L', 'elements': 16, 'id': 41}
    ]},
    {'label': 'Prop', 'detectors': [
        {'name': 'P1X1', 'elements': 72, 'id': 49},
        {'name': 'P1X2', 'elements': 72, 'id': 50}
    ]},
    {'label': 'Hodo', 'detectors': [
        {'name': 'H4Y2R', 'elements': 16, 'id': 44},
        {'name': 'H4Y2L', 'elements': 16, 'id': 43},
        {'name': 'H4T', 'elements': 16, 'id': 46},
        {'name': 'H4B', 'elements': 16, 'id': 45}
    ]},
    {'label': 'Prop', 'detectors': [
        {'name': 'P2X1', 'elements': 72, 'id': 51},
        {'name': 'P2X2', 'elements': 72, 'id': 52},
        {'name': 'P2Y1', 'elements': 72, 'id': 53},
        {'name': 'P2Y2', 'elements': 72, 'id': 54}
    ]}
    ]  
    
    # root_dir = r"/home/ptgroup/Documents/Devin/Big_Data/QTracker_Data/run_005994-20241230T213148Z-001/run_005994"
    root_dir = r"C:\Program Files\Work\QTracker\Root_Files"
    # output_dir = r"/home/ptgroup/Documents/Devin/QTracker/Devin/Data_Viewing/Occupancy_Plots"
    output_dir = r"C:\Program Files\Work\QTracker\Q-Tracker\Devin\Data_Viewing\Occupancy_Plots"
    
    station_data = process_station_occupancy(root_dir, detector_groups)
    plot_station_heatmaps(station_data, detector_groups, output_dir)