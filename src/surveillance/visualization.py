import matplotlib.pyplot as plt
import numpy as np
import logging

class MapVisualizer:
    """
    Handles the real-time visualization of drone maps using Matplotlib.
    """
    def __init__(self, num_drones: int, map_size: int = 100):
        """
        Initializes the plot with a subplot for each drone.

        Args:
            num_drones: The number of drones in the simulation.
            map_size: The total number of cells in the map (e.g., 100 for a 10x10 grid).
        """
        try:
            plt.ion()  # Turn on interactive mode for non-blocking plots
            self.fig, self.axes = plt.subplots(1, num_drones, figsize=(5 * num_drones, 5.5))
            
            # Ensure self.axes is always a list for consistent indexing
            if num_drones == 1:
                self.axes = [self.axes]

            # Calculate the shape of the grid (e.g., 10x10)
            self.map_shape = (int(np.sqrt(map_size)), int(np.sqrt(map_size)))
            
            self.images = []
            for i in range(num_drones):
                # Initialize with a blank map
                initial_map_data = np.zeros(self.map_shape)
                im = self.axes[i].imshow(initial_map_data, cmap='gray_r', vmin=0, vmax=1, origin='lower')
                
                # Set titles and remove axis ticks for a cleaner look
                self.axes[i].set_title(f"Drone {i + 1}'s Map (Initializing...)")
                self.axes[i].set_xticks([])
                self.axes[i].set_yticks([])
                self.images.append(im)
                
            self.fig.tight_layout(pad=2.0)
            plt.show()
        except Exception as e:
            logging.error(f"Error initializing visualizer: {e}")


    def update_map(self, drone_id: int, map_data: np.ndarray):
        """
        Updates the map visualization for a specific drone.

        Args:
            drone_id: The unique ID of the drone from the simulator.
            map_data: The drone's current map data (a NumPy array).
        """
        try:
            # The simulation assigns IDs starting from 100 for the drones.
            # We map this to our plot indices (0, 1, 2).
            plot_index = drone_id - 100 
            
            if 0 <= plot_index < len(self.images):
                # The map is stored as (100, 2), but we only need the first column
                # (the 'seen' status) and reshape it into a 10x10 grid for display.
                map_view = map_data[:, 0].reshape(self.map_shape)
                
                # Update the image data and title
                self.images[plot_index].set_data(map_view)
                discovered_cells = np.sum(map_view)
                self.axes[plot_index].set_title(f"Drone {drone_id}'s Map ({discovered_cells:.0f} / 100)")
                
                # Redraw the canvas to show the changes
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
        except Exception as e:
            # Log errors without crashing the simulation
            logging.warning(f"Could not update map for drone {drone_id}: {e}")
        
    def close(self):
        """Closes the Matplotlib window."""
        plt.ioff()
        plt.close(self.fig)