import random

from gradysim.simulator.handler.mobility import MobilityHandler
from gradysim.simulator.handler.timer import TimerHandler
from gradysim.simulator.handler.visualization import VisualizationHandler
from gradysim.simulator.simulation import SimulationConfiguration, SimulationBuilder
from .inteligent_mobility_protocol import PointOfInterest, Drone
from gradysim.simulator.handler.communication import CommunicationHandler, CommunicationMedium



def main():
    # Configuring simulation
    config = SimulationConfiguration(
        duration=300, 
        real_time=False,
    )
    builder = SimulationBuilder(config)

    builder.add_handler(TimerHandler())
    builder.add_handler(MobilityHandler())
    builder.add_handler(VisualizationHandler())
    builder.add_handler(CommunicationHandler(CommunicationMedium(
        transmission_range=30
    )))

    # Instantiating 4 UAVs at (0,0,0)
    builder.add_node(Drone, (0, 0, 0))
    builder.add_node(Drone, (0, 0, 0))
    #builder.add_node(Drone, (0, 0, 0))
    #builder.add_node(Drone, (0, 0, 0))

    # Instantiating a bunch of ground sensors
    for i in range(10):
        for j in range(10):
            builder.add_node(PointOfInterest,
                             (10*i-50, 10*j-50, 0))

    

    # Building & starting
    simulation = builder.build()
    simulation.start_simulation()


if __name__ == "__main__":
    main()
