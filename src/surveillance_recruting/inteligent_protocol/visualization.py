import matplotlib.pyplot as plt
import numpy as np
import logging

class MapVisualizer:
    """
    Handles the real-time visualization of drone maps using Matplotlib.
    """
    def __init__(self, num_drones: int, map_size: int = 100, threshold: float = 0.5):
        """
        Initializes the plot with a subplot for each drone.

        Args:
            num_drones: The number of drones in the simulation.
            map_size: The total number of cells in the map (e.g., 100 for a 10x10 grid).
        """
        try:
            plt.ion()
            self.fig, self.axes = plt.subplots(2, num_drones, figsize=(5 * num_drones, 10))
            
            if num_drones == 1:
                self.axes = self.axes.reshape(2, 1)

            self.map_shape = (int(np.sqrt(map_size)), int(np.sqrt(map_size)))
            self.threshold = threshold

            self.images = [[], []] 
            
            for i in range(num_drones):
                initial_map_data = np.ones(self.map_shape)

                # --- Setup Top Row (Original Map) ---
                ax_top = self.axes[0, i]
                im_top = ax_top.imshow(initial_map_data, cmap='gray_r', vmin=0, vmax=1, origin='lower')
                ax_top.set_title(f"Drone {i} Map")
                ax_top.set_xticks([])
                ax_top.set_yticks([])
                self.images[0].append(im_top)

                # --- Setup Bottom Row (Masked Map) ---
                ax_bottom = self.axes[1, i]
                im_bottom = ax_bottom.imshow(self.apply_mask(initial_map_data), cmap='RdYlGn', vmin=0, vmax=1, origin='lower')
                ax_bottom.set_title(f"Drone {i} Mask")
                ax_bottom.set_xticks([])
                ax_bottom.set_yticks([])
                self.images[1].append(im_bottom)
                
            self.fig.tight_layout(pad=2.0)
            plt.show()
        except Exception as e:
            logging.error(f"Error initializing visualizer: {e}")

    def apply_mask(self, grid: np.ndarray) -> np.ndarray:
        """Applies a threshold mask to the grid."""
        return (grid <= self.threshold).astype(int)

    def update_map(self, drone_id: int, map_data: np.ndarray):
        """
        Updates the map visualization for a specific drone.
        """
        try:
            # Convert drone ID (100, 101, ...) to plot column index (0, 1, ...)
            plot_index = drone_id
            map_view = map_data.copy()
            
            for i in range(len(map_data)):
                if map_data[i,0] >= 1:
                    map_data[i,0] = 1
            

            if 0 <= plot_index < len(self.images[0]):
                masked_view = self.apply_mask(map_view)

                # Update Top Plot (Original)
                self.images[0][plot_index].set_data(map_view)
                discovered_cells = np.sum(map_view <= self.threshold)
                self.axes[0, plot_index].set_title(f"Drone {drone_id} Map ({discovered_cells:.0f}/100)")
                
                # Update Bottom Plot (Masked)
                self.images[1][plot_index].set_data(masked_view)
                self.axes[1, plot_index].set_title(f"Drone {drone_id} Mask")

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except Exception as e:
            logging.warning(f"Could not update map for drone {drone_id}: {e}")
        
    def close(self):
        """Closes the Matplotlib window."""
        plt.ioff()
        plt.close(self.fig)