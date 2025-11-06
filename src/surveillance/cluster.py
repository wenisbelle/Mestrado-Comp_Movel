import numpy as np
from collections import deque
from typing import Tuple, List
import random
import heapq


class ClusterDetector:
    def __init__(self, threshold: float, min_size: int):
        self.threshold = threshold
        self.min_size = min_size

    def apply_mask(self, grid: np.ndarray) -> np.ndarray:
        """Applies a threshold mask to the grid."""
        return (grid >= self.threshold).astype(int)
    
    def find_zero_clusters(self, grid: np.ndarray) -> list:
    
        if grid is None or grid.size == 0:
            return []
        
        grid = self.apply_mask(grid)

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

                    if len(current_cluster) >= self.min_size:
                        all_clusters.append(current_cluster)

        return all_clusters

    def cluster_fitness(self, all_clusters:list, drone_position: Tuple[float, float, float], map_width: int, map_height: int, distance_norm: int, cluster_size_norm: int) -> list:
        if not all_clusters:
            return []

        drone_x, drone_y, _ = drone_position
        fitness_scores = []

        for cluster in all_clusters:
            center_x = 0
            center_y = 0
            representative_x, representative_y = random.choice(cluster)
            for (row, col) in cluster:
                # Get the center of the cluster
                center_x += col
                center_y += row
            center_x /= len(cluster)
            center_y /= len(cluster)

            map_center_x = center_x * 10 - (map_width * 10) / 2
            map_center_y = center_y * 10 - (map_height * 10) / 2

            #print(f"Cluster center: ({map_center_x}, {map_center_y})")
        
            # Calculate Euclidean distance
            distance = np.sqrt((map_center_x - drone_x) ** 2 + (map_center_y - drone_y) ** 2)/distance_norm # normalize
            cluster_size = len(cluster)/cluster_size_norm

            fitness = cluster_size - distance
            # Store fitness and cluster center coordinates
            fitness_scores.append((fitness, (representative_x, representative_y)))

        return fitness_scores
    
    def choose_one_cluster(self, fitness_scores: list) -> Tuple[float, float]:
        if not fitness_scores:
            return None
        
        if len(fitness_scores) < 3:
            best_cluster = max(fitness_scores, key=lambda x: x[0])
            return best_cluster[1]

        random_sample = random.sample(fitness_scores, 3)
        best_in_sample = max(random_sample, key=lambda x: x[0])
        #print(f"Chosen cluster with fitness: {best_in_sample[0]}")
        # Return the coordinates
        return best_in_sample[1]
    
    def choose_two_clusters(self, fitness_scores: list) ->  List[Tuple[float, float]]:
        if not fitness_scores:
            return None
        
        if len(fitness_scores) == 2:
            return [fitness_scores[0][1], fitness_scores[1][1]]        

        random_sample = random.sample(fitness_scores, 3)
        top_two = heapq.nlargest(2, random_sample, key=lambda x: x[0])
        print(f"Chosen clusters in positions: {[item[1] for item in top_two]}")
        return [item[1] for item in top_two]
