import random
import logging

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .random_protocol import PointOfInterest, drone_protocol_factory
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium

def run_simulation_once(iteration: int):
    # Configuring simulation
    config = SimulationConfiguration(
        duration=250, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    #builder.add_handler(VisualizationHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
    )))    

    drone_params = {
        "uncertainty_rate": 0.05,
        "vanishing_update_time": 10.0,
        "map_threshold": 0.5,
        "distance_norm": 100,
        "cluster_size_norm": 1,
        "number_of_drones": 6,    
        "map_width": 10,
        "map_height": 10
    }

    ConfiguredDrone = drone_protocol_factory(
        uncertainty_rate=drone_params["uncertainty_rate"],
        vanishing_update_time=drone_params["vanishing_update_time"],
        map_threshold=drone_params["map_threshold"],
        distance_norm=drone_params["distance_norm"],
        cluster_size_norm=drone_params["cluster_size_norm"],
        number_of_drones=drone_params["number_of_drones"],
        map_width=drone_params["map_width"],
        map_height=drone_params["map_height"]
    )

    builder.add_node(ConfiguredDrone, (0, 0, 0))
    builder.add_node(ConfiguredDrone, (0, 0, 0))
    builder.add_node(ConfiguredDrone, (0, 0, 0))
    builder.add_node(ConfiguredDrone, (0, 0, 0))
    builder.add_node(ConfiguredDrone, (0, 0, 0))
    builder.add_node(ConfiguredDrone, (0, 0, 0))


    # Instantiating ground sensors to represent the map
    for i in range(10):
        for j in range(10):
            builder.add_node(PointOfInterest,
                             (10*i-50, 10*j-50, 0))    

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()

def main():
    # Configure logging to write to a file
    logging.basicConfig(
        level=logging.INFO,  
        filename=f'/Comp_mov/src/surveillance_v3/logs/article/6UAV_100A/random/simulation.log', 
        filemode='w', 
        #format='%(asctime)s - %(levelname)s - %(message)s'
        format='%(message)s'  
    )
    NUMBER_OF_RUNS = 10
    for i in range(NUMBER_OF_RUNS):
        run_simulation_once(i)


if __name__ == "__main__":
    main()
