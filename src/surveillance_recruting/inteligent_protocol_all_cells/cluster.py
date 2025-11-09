import numpy as np
from collections import deque
from typing import Tuple, List
import random
import heapq


class ClusterDetector:
    def __init__(self, map_width: int, map_height: int):
        self.map_width = map_width
        self.map_height = map_height


    def cells_fitness(self, map_data: np.array, drone_position: Tuple[float, float, float], distance_norm: int, uncertainty_norm: int, map_center_offset: float, distance_between_cells: int) -> list:

        map_data = map_data.copy()
        drone_x, drone_y, _ = drone_position
        fitness_scores = []

        for i, j in np.ndindex(map_data.shape):            
        
            # Calculate Euclidean distance
            x_cell = distance_between_cells*i - map_center_offset
            y_cell = distance_between_cells*j - map_center_offset
            distance = np.sqrt((x_cell - drone_x) ** 2 + (y_cell - drone_y) ** 2)

            #print(f"Cell ({i}, {j}) - Uncertainty: {map_data[i, j]:.4f}, Distance: {distance:.2f}")

            cell_fitness = map_data[i, j]/uncertainty_norm - distance/distance_norm 
            fitness_scores.append((cell_fitness, (i, j)))

        return fitness_scores
    
    def choose_one_cell(self, fitness_scores: list) -> Tuple[float, float]:
        if not fitness_scores:
            return None
        
        top_ten = heapq.nlargest(5, fitness_scores, key=lambda x: x[0])
        random_sample = random.choice(top_ten)
        #best_cell = max(random_sample, key=lambda x: x[0])
        #print(f"Chosen cell with fitness: {best_cell[0]}")
        # Return the coordinates
        return random_sample[1]

    def choose_two_cells(self, fitness_scores: list) ->  List[Tuple[float, float]]:
        if not fitness_scores:
            return None
        
        if len(fitness_scores) == 2:
            best_1 = (fitness_scores[0][1])
            best_2 = (fitness_scores[1][1])
            return [best_1, best_2]
        
        top_five = heapq.nlargest(5, fitness_scores, key=lambda x: x[0])
        
        random_sample = random.sample(top_five,2)
        #top_two = heapq.nlargest(2, random_sample, key=lambda x: x[0])
        #print(f"Chosen clusters in positions: {[item[1] for item in top_two]}")
        best_1 = (random_sample[0][1])
        best_2 = (random_sample[1][1])
        return [best_1, best_2]
