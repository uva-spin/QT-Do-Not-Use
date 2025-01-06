import numpy as np
import matplotlib.pyplot as plt


def Background():
    rows, cols = 201, 62

    matrix = np.zeros((rows, cols), dtype=int)

    # Parameters for outages, individual hits, and noise
    num_stacks = 5  
    max_stack_height = 10  
    cluster_radius = 30 
    cluster_intensity = 75  # number of hits in cluster

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
    ]
    
    # Updated probabilities (now has 12 values, one for each interval)
    probabilities = [0.3, 0.05, 0.2, 0.1, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0]  # Set probability for 56-61 to 0

    # Ensure the probabilities add up to 1
    probabilities = np.array(probabilities) / np.sum(probabilities)

    # Step 1: Add random vertical stacks within each of the ranges
    for _ in range(num_stacks):
        interval_idx = np.random.choice(len(intervals), p=probabilities)
        col_start, col_end = intervals[interval_idx][0], intervals[interval_idx][1]

        col = np.random.randint(col_start, col_end + 1)

        start_row = np.random.randint(0, rows - max_stack_height + 1)

        stack_height = np.random.randint(1, max_stack_height + 1)

        matrix[start_row:start_row + stack_height, col] = 1

    # Step 2: Add random individual hits within each of the ranges
    for col_start, col_end, max_element, num_random_hits in intervals:
        if col_start == 56 and col_end == 61:
            # Skip this interval and do not add any hits
            continue

        # Generate random row indices and column indices for the hits
        random_positions = np.random.choice(rows * (col_end - col_start + 1), num_random_hits, replace=False)
        random_row_indices, random_col_indices = np.unravel_index(random_positions, (rows, col_end - col_start + 1))

        # Adjust column indices to fit into the specified range
        random_col_indices += col_start

        # Ensure that row indices don't fall within the forbidden range (56-61)
        valid_row_indices = random_row_indices[
            (random_row_indices < 56) | (random_row_indices > 61)
        ]
        valid_col_indices = random_col_indices[:len(valid_row_indices)]

        # Distribute hits across the element IDs
        valid_row_indices = np.random.randint(0, max_element, size=valid_row_indices.shape)

        for row, col in zip(valid_row_indices, valid_col_indices):
            if matrix[row, col] == 0:  # Avoid overriding existing hits
                matrix[row, col] = 1

    # Step 3: Add Gaussian circular cluster of hits (noise) in a random location
    center_row = np.random.randint(cluster_radius, rows - cluster_radius)
    center_col = np.random.randint(cluster_radius, cols - cluster_radius)

    for _ in range(cluster_intensity):
        r = np.random.normal(0, cluster_radius / 3)
        theta = np.random.uniform(0, 2 * np.pi)
        hit_row = int(center_row + r * np.sin(theta))
        hit_col = int(center_col + r * np.cos(theta))

        if 0 <= hit_row < rows and 0 <= hit_col < cols:
            matrix[hit_row, hit_col] = 1

    row_indices, col_indices = np.nonzero(matrix)
    return col_indices, row_indices


col_indices, row_indices = Background()

plt.figure(figsize=(10, 6))
plt.scatter(col_indices, row_indices, c='black', marker='_')

# Set the title and labels
plt.title(f'Background Hit Matrix')
plt.xlabel('Detector ID')
plt.ylabel('Element ID')

# Adjust the x-axis ticks to start and end at the corners of the grid (0 to 61)
plt.xticks(np.arange(0, 62, 4))  # Set x-axis ticks from 0 to 61
plt.xlim(0, 61)


plt.savefig("Background_Noise_Example.jpeg")



