import os
import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

def read_event(file_path, event_number):
    with uproot.open(file_path + ":save") as file:
        detectorid = file["fAllHits.detectorID"].array(library="np")[event_number]
        elementid = file["fAllHits.elementID"].array(library="np")[event_number]
    return detectorid, elementid

def create_plot(file_paths):
    """Create and display a plot with the data accumulated from all ROOT files."""
    fig, ax = plt.subplots(figsize=(16, 8))
    y_max = max([det['elements'] for group in detector_groups for det in group['detectors']])  # Maximum number of elements for scaling
    x_labels = []
    x_ticks = []
    x_offset = 0

    # Loop through all ROOT files and events
    for file_path in file_paths:
        with uproot.open(file_path + ":save") as file:
            total_events = len(file["fAllHits.detectorID"].array(library="np"))
            for event_number in range(total_events):
                print(f"Processing event {event_number} from file {file_path}")
                detectorid, elementid = read_event(file_path, event_number)

                # Plot each group with spacing between them
                for group in detector_groups:
                    x_positions = range(x_offset, x_offset + len(group['detectors']))
                    for idx, x in enumerate(x_positions):
                        detector = group['detectors'][idx]
                        y_positions = range(detector['elements'])

                        for y in y_positions:
                            # Scale y to match the maximum elements
                            scaled_y = y * y_max / detector['elements']
                            height = y_max / detector['elements']

                            # Check if the element is a hit
                            is_hit = any((detectorid == detector['id']) & (elementid == (y + 1)))
                            color = 'yellow' if is_hit else 'darkblue'

                            ax.add_patch(Rectangle((x, scaled_y), 1, height, edgecolor='black', facecolor=color))

                        # Add x-axis labels
                        x_labels.append(detector['name'])
                        x_ticks.append(x + 0.5)

                    # Label the group
                    ax.text(x_offset + len(group['detectors']) / 2, y_max + 15, group['label'], ha='center', va='center')

                    # Add spacing between groups
                    x_offset += len(group['detectors']) + 2

    # Set the limits and labels
    ax.set_xlim(0, x_offset)
    ax.set_ylim(0, y_max + 20)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=90)
    ax.set_xlabel('')
    ax.set_ylabel('Element ID')

    plt.tight_layout()
    plt.show()

def main(directory):
    """Main function to collect files and create the plot."""
    files = [f for f in os.listdir(directory) if f.endswith('.root')]
    if not files:
        print("No ROOT files found in the directory.")
        return

    # Sort files by creation time (most recent first)
    sorted_files = sorted([os.path.join(directory, f) for f in files], key=os.path.getctime, reverse=True)

    # Call the create_plot function with the list of ROOT files
    create_plot(sorted_files)

if __name__ == "__main__":
    directory = r'/home/ptgroup/Documents/Devin/Big_Data/QTracker_Data/run_005994-20241230T213148Z-001/run_005994'  # Set the directory path where your ROOT files are located
    main(directory)
