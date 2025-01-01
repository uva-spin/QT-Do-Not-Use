import os
import numpy as np
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def accumulate_occupancy_by_detector(root_dir, detector_groups, debug=False):
  """
  Accumulates occupancy for each detector within the specified groups.

  Args:
      root_dir: Path to the directory containing ROOT files.
      detector_groups: List of dictionaries, where each dictionary represents a detector group 
                       and contains a 'label' and a list of 'detectors'.
      debug: Boolean flag to enable debugging messages.

  Returns:
      A dictionary where keys are station names and values are dictionaries mapping
      detector IDs to their occupancy counts.
  """

  occupancies = {}
  for group in detector_groups:
    occupancies[group['label']] = {}

  for filename in os.listdir(root_dir):
    if filename.endswith(".root"):
      file_path = os.path.join(root_dir, filename)
      print(f"Processing file: {file_path}")

      try:
        with uproot.open(file_path + ":save") as file:
          detectorid = file["fAllHits.detectorID"].array(library="np")
          elementid = file["fAllHits.elementID"].array(library="np")
      except Exception as e:
        if debug:
          print(f"Error reading file {file_path}: {e}")
        continue

      for group in detector_groups:
        for detector in group['detectors']:
          mask = (detectorid == detector['id'])
          occupancies[group['label']][detector['id']] = np.sum(mask & (len(elementid) > 0))

  return occupancies

def plot_occupancy_heatmap(occupancies, detector_groups, output_dir):
  """
  Plots a heatmap representing the occupancy for each detector within groups.

  Args:
      occupancies: Dictionary where keys are station names and values are dictionaries
                   mapping detector IDs to their occupancy counts.
      detector_groups: Dictionary mapping station names to lists of detector properties.
      output_dir: Path to the directory for saving the heatmap.
  """

  y_max = max([det['elements'] for group in detector_groups for det in group['detectors']])

  fig, ax = plt.subplots()

  # Define colormap for occupancy visualization
  cmap = plt.cm.YlOrBr

  # Loop through detector groups
  x_offset = 0
  for group in detector_groups:
    x_positions = range(x_offset, x_offset + len(group['detectors']))

    # Plot heatmap for each detector within the group
    for idx, detector in enumerate(group['detectors']):
      detector_id = detector['id']
      occupancy = occupancies.get(group['label'], {}).get(detector_id, 0)
      y_positions = range(detector['elements'])

      # Calculate scaled occupancy for colormap
      scaled_occupancy = occupancy / (y_max * detector['elements'])

      # Create heatmap image data
      heatmap_data = np.zeros((detector['elements'], 1))
      heatmap_data[:, 0] = scaled_occupancy

      # Plot the heatmap
      ax.imshow(heatmap_data, extent=(x_positions[idx], x_positions[idx] + 1, 0, y_max), cmap=cmap)

    # Add group label
    ax.text(x_offset + len(group['detectors']) / 2, y_max + 5, group['label'], ha='center', va='center', fontsize=12)

    # Update x-axis offset for next group
    x_offset += len(group['detectors']) + 2

  # Set labels and ticks
  plt.xlabel('Detector Name', fontsize=12)
  plt.ylabel('Element ID', fontsize=12)
  plt.xticks(range(len(detector_groups)), [group['label'] for group in detector_groups], rotation=45, ha='right', fontsize=12)
  plt.colorbar(label='Occupancy (Normalized)', orientation='vertical', fraction=0.05)

  # Adjust layout and save the plot
  plt.tight_layout()
  plt.savefig(os.path.join(output_dir, "occupancy_heatmap.png"))
  plt.close(fig)

if __name__ == "__main__":
  # Define detector groups (Stations 1, 2, 3+, and 3-)
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

  # Specify input and output directories
  root_dir = r"/home/ptgroup/Documents/Devin/Big_Data/QTracker_Data/run_005994-20241230T213148Z-001/run_005994" 
  output_dir = r"/home/ptgroup/Documents/Devin/QTracker/Devin/Data_Viewing/Occupancy_Plots"

  # Enable debugging messages
  debug_mode = True 

  # Accumulate occupancy for each station
  occupancies = accumulate_occupancy_by_detector(root_dir, detector_groups, debug=debug_mode)

  # Plot and save occupancy heatmap
  plot_occupancy_heatmap(occupancies, detector_groups, output_dir)