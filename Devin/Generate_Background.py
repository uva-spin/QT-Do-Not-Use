import numpy as np
import matplotlib.pyplot as plt

rows, cols = 201, 61

matrix = np.zeros((rows, cols), dtype=int)

num_stacks = 5  
max_stack_height = 10  
num_random_hits = 10

# Step 1: Add random vertical stacks
for _ in range(num_stacks):

    col = np.random.randint(0, cols)
    
    start_row = np.random.randint(0, rows - max_stack_height + 1)
    
    stack_height = np.random.randint(1, max_stack_height + 1)
    
    matrix[start_row:start_row + stack_height, col] = 1

# Step 2: Add random individual hits
random_positions = np.random.choice(rows * cols, num_random_hits, replace=False)
random_row_indices, random_col_indices = np.unravel_index(random_positions, (rows, cols))

for row, col in zip(random_row_indices, random_col_indices):
    if matrix[row, col] == 0: 
        matrix[row, col] = 1

# Step 3: Extract row and column indices of all the hits
row_indices, col_indices = np.nonzero(matrix)

plt.figure(figsize=(10, 5))
plt.scatter(col_indices, row_indices, c='black', marker='_')
plt.gca().invert_yaxis()  
plt.title(f'Background Hit Matrix')
plt.xlabel('Element ID')
plt.ylabel('Detector ID')
plt.legend()
# plt.grid(True)
# plt.show()
plt.savefig("Background_Noise_Example.jpeg")
