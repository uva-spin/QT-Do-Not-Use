"""
This code is used to generate a background data for training DNNs. 

The background is generated by adding random vertical stacks of hits, random individual hits, and a Gaussian cluster of hits.

- Random vertical stacks of hits: This is to simulate the effect of voltage surges in the detector (somewhat rare, but happens typically in the same places).
- Random individual hits: This is to simulate the effect of ionization in the drift chamber from cosmic rays (rare, but happens).
- Gaussian cluster of hits: This is to simulate the effect of ionization in the drift chamber from muon tracks.

The background is then added to the hit matrix of the ideal events.

Keep in mind that a MAJORITY of the noise we see in data events is due to partial muon tracks (ones that don't go through the entire detector, up to some cut-off).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from matplotlib.patches import Patch

from Dimuon_Sim import DataProcessing

class Background:
    def __init__(self, hit_matrix_mup, hit_matrix_mum):
        self.hit_matrix_mup = hit_matrix_mup
        self.hit_matrix_mum = hit_matrix_mum

    def generate_background(self):
        rows, cols = 201, 62 ### This is the size of the hit matrix

        matrix = np.zeros((rows, cols), dtype=int)

        # Parameters for outages, individual hits, and noise
        num_stacks = 2  ### This is the number of vertical stacks of hits, typically from voltage surges
        max_stack_height = 10  ### This is the maximum height of a stack
        cluster_radius = 5  ### This is the radius of the Gaussian cluster of hits, typically from ionization in drift chamber
        cluster_intensity = 10  ### This is the number of hits in the cluster

        # Define the vertical ranges along the x-axis and their respective hit distributions
        intervals = [
            (0, 5, 190, 100),   # 0-5 (more hits, max element ID 190)
            (6, 13, 50, 50),    # 6-13 (equal hits, max element ID around 75)
            (14, 19, 75, 100),  # 14-19 (more hits, max element ID around 75)
            (20, 27, 30, 50),   # 20-27 (equal hits)
            (28, 39, 100, 100), # 28-39 (more hits, max element ID 100)
            (40, 41, 30, 30),   # 40-41 (equal hits)
            (42, 43, 50, 75),   # 42-43 (less hits, max element ID around 75)
            (46, 47, 20, 30),   # 46-47 (equal hits)
            (48, 49, 30, 30),   # 48-49 (equal hits)
            (50, 53, 25, 50),   # 50-53 (equal hits)
            (54, 55, 15, 30),   # 54-55 (equal hits)
            (56, 61, 10, 0)     # 56-61 (empty, no hits)
        ]  ### This is all just hand-wavy, we need to compare against the data to see what is actually happening
        
        ### This is the probability of each interval to be chosen
        probabilities = [0.3, 0.05, 0.2, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0]  # Set probability for 56-61 to 0

        ### Ensure the probabilities add up to 1
        probabilities = np.array(probabilities) / np.sum(probabilities)

        # Step 1: Add random vertical stacks within each of the ranges
        for _ in range(num_stacks):
            interval_idx = np.random.choice(len(intervals), p=probabilities)
            col_start, col_end = intervals[interval_idx][0], intervals[interval_idx][1]

            col = np.random.randint(col_start, col_end + 1)

            start_row = np.random.randint(0, rows - max_stack_height + 1)

            stack_height = np.random.randint(1, max_stack_height + 1)

            matrix[start_row:start_row + stack_height, col] = 1
            
        ### This needs to be worked on below ###

        # # Step 2: Add random individual hits within each of the ranges
        # for col_start, col_end, max_element, num_random_hits in intervals:
        #     if col_start == 56 and col_end == 61:
        #         # Skip this interval and do not add any hits
        #         continue

        #     # Generate random row indices and column indices for the hits
        #     random_positions = np.random.choice(rows * (col_end - col_start + 1), num_random_hits, replace=False)
        #     random_row_indices, random_col_indices = np.unravel_index(random_positions, (rows, col_end - col_start + 1))

        #     # Adjust column indices to fit into the specified range
        #     random_col_indices += col_start

        #     # Ensure that row indices don't fall within the forbidden range (56-61)
        #     valid_row_indices = random_row_indices[
        #         (random_row_indices < 56) | (random_row_indices > 61)
        #     ]
        #     valid_col_indices = random_col_indices[:len(valid_row_indices)]

        #     # Distribute hits across the element IDs
        #     valid_row_indices = np.random.randint(0, max_element, size=valid_row_indices.shape)

        #     for row, col in zip(valid_row_indices, valid_col_indices):
        #         if matrix[row, col] == 0:  # Avoid overriding existing hits
        #             matrix[row, col] = 1


        # Step 3: Add Gaussian circular cluster of hits (noise) in a random location
        center_row = np.random.randint(cluster_radius, rows - cluster_radius)
        center_col = np.random.randint(cluster_radius, cols - cluster_radius)

        for _ in range(cluster_intensity):
            r = np.random.normal(0, cluster_radius)
            theta = np.random.uniform(0, 2 * np.pi)
            hit_row = int(center_row + r * np.sin(theta))
            hit_col = int(center_col + r * np.cos(theta))

            if 0 <= hit_row < rows and 0 <= hit_col < cols:
                matrix[hit_row, hit_col] = 1
                
        # Step 4: Add partial tracks (Most of the noise is due to partial tracks)
        
        # Sum over events to get 2D hit matrices (this include the occupancy of the hits)
        hit_matrix_mup_2d = np.sum(hit_matrix_mup, axis=0)  # Shape (62, 201)
        hit_matrix_mum_2d = np.sum(hit_matrix_mum, axis=0)  # Shape (62, 201)

        # Add the hit matrices to the background matrix
        matrix += hit_matrix_mup_2d.transpose()
        matrix += hit_matrix_mum_2d.transpose()

        return matrix, hit_matrix_mup_2d, hit_matrix_mum_2d
    
    def plot_hits(self,ax, hit_matrix, color: str, label: str):
        """
        Plots individual hits

        Args:
            ax: Matplotlib axis object.
            hit_matrix (np.ndarray): 2D array of hit data.
            color (str): Color for the hits.
            label (str): Label for the legend.
        """
        
        print(hit_matrix.shape)
        y, x = np.where(hit_matrix.T > 0)  # Transpose for correct orientation
        sns.scatterplot(x=x, y=y, color=color, label=label, marker='_', s=100, alpha=0.8, ax=ax)

    def plot_heatmap(self,ax, hit_matrix, cmap: str, alpha: float = 0.7):
        """
        Plots a heatmap of hit matrix

        Args:
            ax: Matplotlib axis object.
            hit_matrix (np.ndarray): 2D array of hit data.
            cmap (str): Colormap for the heatmap.
            alpha (float): Transparency level.
        """
        sns.heatmap(hit_matrix, cmap=cmap, ax=ax, alpha=alpha, cbar=False)
    
    def visualize_background(self,matrix, hit_matrix_mup_2d, hit_matrix_mum_2d, plot_mode: str = "heatmap"):
        """
        Visualizes the hit matrices for muon plus and muon minus tracks using Seaborn styling.
        Args:
            hit_matrix_mup (np.ndarray): 3D hit matrix for muon plus (num_events, 62, 201).
            hit_matrix_mum (np.ndarray): 3D hit matrix for muon minus (num_events, 62, 201).
            plot_mode (str): Visualization mode ("hits" or "heatmap").
        """
        sns.set_style("ticks")
        sns.set_theme(style="darkgrid", palette="deep")
        sns.set_context("notebook", font_scale=1.2)
        
        fig, ax = plt.subplots(figsize=(16, 8))

        if plot_mode == "heatmap":
            # Heatmap mode
            self.plot_heatmap(ax, matrix, cmap='magma', alpha=0.9)
            self.plot_heatmap(ax, hit_matrix_mup_2d, cmap='Reds', alpha=0.9)
            self.plot_heatmap(ax, hit_matrix_mum_2d, cmap='Blues', alpha=0.9)

            # Create proxy artists for the legend 
            legend_elements = [
                Patch(facecolor=sns.color_palette("magma")[-1], label='Total Background', alpha=0.9),
                Patch(facecolor=sns.color_palette("Reds")[-1], label='Muon Plus', alpha=0.9),
                Patch(facecolor=sns.color_palette("Blues")[-1], label='Muon Minus', alpha=0.9),
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12, title="Track Type", 
                    title_fontsize=14, frameon=True, fancybox=True, shadow=True)

        elif plot_mode == "hits":
            # Hits mode: Plot individual points 
            self.plot_hits(ax, hit_matrix_mup, color=sns.color_palette("Reds")[-1], label='Muon Plus')
            self.plot_hits(ax, hit_matrix_mum, color=sns.color_palette("Blues")[-1], label='Muon Minus')
            self.plot_hits(ax, matrix, color=sns.color_palette("magma")[-1], label='Total Background')
            # ax.invert_yaxis()  


            ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)

        ax.set_xlabel("Detector ID", fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel("Element ID", fontsize=14, fontweight='bold', labelpad=10)
        ax.set_title("Overlay of Muon Tracks", fontsize=16, fontweight='bold', pad=20)
        
        # Set tick positions and labels
        ax.set_xticks(np.arange(0, 62, 2))  
        ax.set_xticklabels(np.arange(0, 62, 2)) 
        ax.set_yticks(np.arange(0, 201, 20)) 
        ax.set_yticklabels(np.arange(0, 201, 20)) 
        
        plt.xticks(rotation=45, ha='right') 
        # plt.yticks(rotation=45, ha='right')
        
        ax.invert_yaxis()  
        ax.set_aspect(0.1) 
        
        # Add grid with Seaborn styling
        # ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        script_dir = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(script_dir, "Background_Noise_Example.jpeg"), dpi=300)
        plt.show()


if __name__ == "__main__":
        ### Load root files and get hit matrices
    root_file = "Devin/Mock_Dimuon/Data_Files/Dimuon_target_100K.root"
    max_events = 60000  # Number of events to process
    dp = DataProcessing(root_file)

    # Get total number of events in the file
    num_events = dp.get_num_events()

    # Generate hit matrices using only selected ideal events
    ideal_events = [event for event in range(dp.get_num_events()) if dp.find_ideal_events(event)]
    selected_events = ideal_events[:max_events]
    
    # Select up to `max_events` ideal events
    selected_events = ideal_events[:max_events]
    print(f"There are {len(selected_events)} selected ideal events.")

    # If no ideal events are found, exit early
    if len(selected_events) == 0:
        print("No ideal events found. Exiting.")
        exit()

    # Generate hit matrices using only selected ideal events
    (
        truth_elementID_mup,  
        truth_elementID_mum,  
        truth_values_drift_mup,  
        truth_values_drift_mum,  
        hit_matrix_mup,  ### We interested in this
        hit_matrix_mum   ### And this
        
    ) = dp.make_hit_matrix(selected_events, quality_metric=1.0)
    
    print(hit_matrix_mup)
    print(hit_matrix_mum)
    
    ### Initialize background generator
    
    Background_Generator = Background(hit_matrix_mup, hit_matrix_mum)
    matrix, hit_matrix_mup_2d, hit_matrix_mum_2d = Background_Generator.generate_background()
    
    Background_Generator.visualize_background(matrix, hit_matrix_mup_2d, hit_matrix_mum_2d, plot_mode="heapmap")



