import numpy as np
from collections import deque

def find_zero_clusters(grid: np.ndarray) -> list:
    """
    Finds and clusters all contiguous regions of zeros in a 2D grid.

    Args:
        grid: A 2D NumPy array representing the occupancy grid (0s and 1s).

    Returns:
        A list of lists, where each inner list contains the (row, col)
        coordinates of a single cluster of zeros.
    """
    if grid is None or grid.size == 0:
        return []

    rows, cols = grid.shape
    visited = np.full((rows, cols), False)
    all_clusters = []

    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and not visited[r, c]:
                current_cluster = []
                queue = deque([(r, c)])
                visited[r, c] = True
                
                while queue:
                    curr_r, curr_c = queue.popleft()
                    current_cluster.append((curr_r, curr_c))

                    # Check all 4 neighbors (up, down, left, right)
                    for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        next_r, next_c = curr_r + dr, curr_c + dc

                        # Check if it's valid
                        if 0 <= next_r < rows and 0 <= next_c < cols and \
                           not visited[next_r, next_c] and grid[next_r, next_c] == 0:
                            
                            visited[next_r, next_c] = True
                            queue.append((next_r, next_c))
                
                all_clusters.append(current_cluster)

    return all_clusters


occupancy_grid = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0]
])

clusters = find_zero_clusters(occupancy_grid)

# Print the results
print("Found clusters:")
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")