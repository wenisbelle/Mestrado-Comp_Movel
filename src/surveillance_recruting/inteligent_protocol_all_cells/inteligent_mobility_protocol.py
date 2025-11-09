import enum
import logging
from typing import TypedDict, Type
import numpy as np
from dataclasses import dataclass
import json
import random
from .visualization import MapVisualizer
from .cluster import ClusterDetector

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand
from gradysim.simulator.extension.camera import CameraHardware, CameraConfiguration
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand



@dataclass
class Threat:
    level: int
    position_id: int
    timestamp: float 

class DroneStatus(enum.Enum):
    MAPPING = 0
    RECRUITING = 1
    ENGAGING = 2

class MessageType(enum.Enum):
    HEARTBEAT_MESSAGE = 0
    SHARE_MAP_MESSAGE = 1
    SHARE_GOTO_POSITION_MESSAGE = 2
    RECRUITING_MESSAGE = 3

class HeartBeatMessage(TypedDict):
    message_type: int
    status: int
    sender: int

class ShareMapMessage(TypedDict):
    message_type: int 
    map: list
    sender: int

class SendGoToMessage(TypedDict):
    message_type: int 
    goto: list
    sender: int

class PointOfInterest(IProtocol):
    is_threat: int

    def initialize(self) -> None:
        self.is_threat = 1
        #logging.info(f"The cell is {self.is_threat}")

    def handle_timer(self, timer: str) -> None:
        pass

    def handle_packet(self, message: str) -> None:
        pass

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        pass


class Drone(IProtocol):
    camera: CameraHardware
    _log: logging.Logger
    visualizer: MapVisualizer = None

    _config = {
        "uncertainty_rate": 10,
        "vanishing_update_time": 10.0,
        "map_threshold": 10,
        "distance_norm": 100,
        "cluster_size_norm": 10,
        "number_of_drones": 2,
        "map_width": 10,
        "map_height": 10
    }

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.drone_position = None
        self.goto_command = np.zeros(3)

        self.UNCERTAINTY_RATE = self._config["uncertainty_rate"]
        self.VANISHING_UPDATE_TIME = self._config["vanishing_update_time"]
        self.MAP_THRESHOLD = self._config["map_threshold"]
        self.DISTANCE_NORM = self._config["distance_norm"]
        self.CLUSTER_SIZE_NORM = self._config["cluster_size_norm"]
        self.NUMBER_OF_DRONES = self._config["number_of_drones"]
        self.MAP_WIDTH = self._config["map_width"]
        self.MAP_HEIGHT = self._config["map_height"]        

        
        ## Initialize map
        self.map = np.zeros((self.MAP_HEIGHT, self.MAP_WIDTH, 2))
        self.map[:,:,0] = 1
        self.total_uncertainty = self.map[:,:,0].sum()
        self.is_cell_visited = np.zeros((self.MAP_HEIGHT, self.MAP_WIDTH))

        self.threats_found = []
        
        ## Initial state
        self.status = DroneStatus.MAPPING      
        

        self._log.info(f"Drone {self.provider.get_id()} uncertainty rate {self.UNCERTAINTY_RATE}, vanishing time {self.VANISHING_UPDATE_TIME}, map threshold {self.MAP_THRESHOLD}")

        ## Visualization and cluster plugins
        self.cluster_detector = ClusterDetector(map_width=self.MAP_WIDTH, map_height=self.MAP_HEIGHT)
        self.last_drone_interaction_time = np.zeros(self.NUMBER_OF_DRONES)  

        #if Drone.visualizer is None:
            # We have 3 drones in the simulation.
            # I have to think a better way to do this.
        #    Drone.visualizer = MapVisualizer(num_drones=self.NUMBER_OF_DRONES, map_size=self.MAP_WIDTH * self.MAP_HEIGHT, threshold=self.MAP_THRESHOLD)

        #self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])

        self.goto_command = np.array([random.uniform(-5*self.MAP_WIDTH, 5*self.MAP_WIDTH), random.uniform(-5*self.MAP_HEIGHT, 5*self.MAP_HEIGHT), 10])
        command = GotoCoordsMobilityCommand(*self.goto_command)
        self.provider.send_mobility_command(command)
        
        ## First callbacks
        self.provider.schedule_timer("mobility",self.provider.current_time() + 5)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 1)
        self.vanishing_map_routine()
        self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)

        ## Camera Configuration
        configuration = CameraConfiguration(20,30,180,0)
        self.camera = CameraHardware(self, configuration)

    def camera_routine(self):
        detected_nodes = self.camera.take_picture()
        for node in detected_nodes:
            #logging.info(f"Detected point of interest at {node['position']} and type {node['type']}") # saida da camera me da o id do node
            flat_index = node['type'] - self.NUMBER_OF_DRONES
            if flat_index < self.MAP_WIDTH * self.MAP_HEIGHT and flat_index >= 0:
                row = flat_index // self.MAP_WIDTH
                col = flat_index % self.MAP_WIDTH

                self.map[row, col, 0] = 0.0 
                self.map[row, col, 1] = self.provider.current_time()
                                
                # Checking the total uncertainty
                self.total_uncertainty = self.map[:,:,0].sum()
                self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has total uncertainty of {self.total_uncertainty}")         

        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])

    def vanishing_map_routine(self):
        self.map[:, :, 0] = self.map[:, :, 0] + self.UNCERTAINTY_RATE
        ## Cheching the number of cells abouve the threshold
        discovered_cells = np.sum(self.map[:, :, 0] <= self.MAP_THRESHOLD)
        self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a total of {discovered_cells} cell abouve threshold")

        ## Checking if the cell was visited
        self.is_cell_visited[self.map[:, :, 1] > 0.0] = 1
        self._log.info(f"At time: {self.provider.current_time()}, the node {self.provider.get_id()} has {self.MAP_WIDTH*self.MAP_HEIGHT - np.sum(self.is_cell_visited)} unvisited cells")

        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])
        

    def internal_mobility_command(self):
        map_center_offset = (self.MAP_WIDTH * 10) / 2

        cells_fitness_scores = self.cluster_detector.cells_fitness(
            self.map[:, :, 0],
            self.drone_position, 
            distance_norm=self.MAP_WIDTH*self.MAP_HEIGHT, 
            uncertainty_norm=1,
            map_center_offset=map_center_offset,
            distance_between_cells=10
        )

        target_coords = self.cluster_detector.choose_one_cell(cells_fitness_scores)
        target_row, target_col = target_coords

        x_goto = target_row * 10 - map_center_offset
        y_goto = target_col * 10 - map_center_offset            
        self.goto_command = np.array([x_goto, y_goto, 10])        
  
        command = GotoCoordsMobilityCommand(*self.goto_command)      
        self.provider.send_mobility_command(command)

    def external_mobility_command(self):
        map_center_offset = (self.MAP_WIDTH * 10) / 2

        cells_fitness_scores = self.cluster_detector.cells_fitness(
            self.map[:, :, 0],
            self.drone_position, 
            distance_norm=self.MAP_WIDTH*self.MAP_HEIGHT, 
            uncertainty_norm=1,
            map_center_offset=map_center_offset,
            distance_between_cells=10
        )

        target_coords = self.cluster_detector.choose_two_cells(cells_fitness_scores)
        target_row_1, target_col_1 = target_coords[0]
        target_row_2, target_col_2 = target_coords[1]

        x_goto = target_row_1 * 10 - map_center_offset
        y_goto = target_col_1 * 10 - map_center_offset            
        self.goto_command = np.array([x_goto, y_goto, 10])

        x_send_command = target_row_2 * 10 - map_center_offset
        y_send_command = target_col_2 * 10 - map_center_offset
        send_command = np.array([x_send_command, y_send_command, 10])

        return send_command    

    
    def send_heartbeat(self):
        #self._log.info(f"Sending heartbeat ...")
        message: HeartBeatMessage = {
            'message_type': MessageType.HEARTBEAT_MESSAGE.value,
            'status': self.status.value,
            'sender': self.provider.get_id()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def send_goto_command(self, send_command: np.array, destination_id: int):
        message: SendGoToMessage = {
            'message_type': MessageType.SHARE_GOTO_POSITION_MESSAGE.value,
            'goto': send_command.tolist(),
            'sender': self.provider.get_id()
        }
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)


    def compare_maps(self, incoming_map: np.ndarray) -> np.ndarray:
        condition = incoming_map[:, :, 1] > self.map[:, :, 1]
        condition_3d = condition[..., np.newaxis]
        return np.where(condition_3d, incoming_map, self.map)
    
    def check_maximum_cells(self):
        #self._log.info(f"Minimum cell value is {self.map[:,0].min()}")
        max_knowledge = self.map[:, :, 0].max()
        rows, cols = np.where(self.map[:, :, 0] == max_knowledge)
        return list(zip(rows, cols))

    def received_heartbeat(self, data: dict):
        heartbeat_msg: HeartBeatMessage = data
        #self._log.info(f"Received heartbeat from {heartbeat_msg['sender']}")

        if heartbeat_msg['status'] == DroneStatus.MAPPING.value and self.status == DroneStatus.MAPPING:
            message: ShareMapMessage = {
                'message_type': MessageType.SHARE_MAP_MESSAGE.value,
                'map': self.map.tolist(),
                'sender': self.provider.get_id()
                }
            destination_id = heartbeat_msg['sender']                
            command = SendMessageCommand(json.dumps(message), destination_id)
            self.provider.send_communication_command(command)


    def updated_map(self, data: dict):
        share_map_msg: ShareMapMessage = data
        updated_map = self.compare_maps(np.array(share_map_msg['map']))
        
        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])

        return updated_map
        
    def handle_timer(self, timer: str) -> None:
        if timer == "camera":
            self.camera_routine()
            self.provider.schedule_timer("camera", self.provider.current_time() + 0.5)
        
        if timer == "mobility": 
            if self.drone_position is not None:
                current_pos_array = np.array(self.drone_position)    
            
            if np.linalg.norm(current_pos_array - self.goto_command) < 1:
                self.internal_mobility_command()

            self.provider.schedule_timer(
                "mobility",
                self.provider.current_time() + 5
            )

        if timer == "heartbeat":
            self.send_heartbeat()
            self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)

        if timer == "vanishing_map":
            self.vanishing_map_routine()
            self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)
            

    def handle_packet(self, message: str) -> None:
        data: dict = json.loads(message)

        if 'message_type' not in data:
           self._log.warning(f"Received message without a message_type: {data}")
           return
        
        msg_type = data['message_type']

        if msg_type == MessageType.HEARTBEAT_MESSAGE.value:
            self.received_heartbeat(data)

        elif msg_type == MessageType.SHARE_MAP_MESSAGE.value:
            self.map = self.updated_map(data)

            if self.provider.current_time() - self.last_drone_interaction_time[data['sender']]  > 3.0: # the drone id starts at 0
                if self.provider.get_id() >= data['sender']:
                    #self._log.info(f"Node {self.provider.get_id()} is calculating the new destinations")
                    send_command = self.external_mobility_command()

                    #self._log.info(f"After updating map going to {self.goto_command} and sending {send_command} to drone {data['sender']}")
                    command = GotoCoordsMobilityCommand(*self.goto_command)
                    self.provider.send_mobility_command(command)

                    self.send_goto_command(send_command, data['sender'])
                self.last_drone_interaction_time[data['sender']] = self.provider.current_time() # the drone id starts at 0
        
        elif msg_type == MessageType.SHARE_GOTO_POSITION_MESSAGE.value:
            goto_msg: SendGoToMessage = data
            #self._log.info(f"Received goto command from {goto_msg['sender']}. Going to {goto_msg['goto']}")
            self.goto_command = goto_msg['goto']
            command = GotoCoordsMobilityCommand(*self.goto_command)      
            self.provider.send_mobility_command(command)       
        
        else:
            self._log.warning(f"Received message with unknown type: {msg_type}")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.drone_position = telemetry.current_position


    def finish(self) -> None:
        logging.info(f"Drone: {self.provider.get_id()}. Final map: {self.map}")

def drone_protocol_factory(
    uncertainty_rate: float, 
    vanishing_update_time: float, 
    map_threshold: float,
    distance_norm: float,
    cluster_size_norm: float,
    number_of_drones: int,
    map_width: int,
    map_height: int
) -> Type[Drone]:
    """
    Creates a new Drone protocol class with the specified configuration.
    """
    # Create a new configuration dictionary
    config = {
        "uncertainty_rate": uncertainty_rate,
        "vanishing_update_time": vanishing_update_time,
        "map_threshold": map_threshold,
        "distance_norm": distance_norm,
        "cluster_size_norm": cluster_size_norm,
        "number_of_drones": number_of_drones,
        "map_width": map_width,
        "map_height": map_height
    }

    # Define a new class that inherits from Drone
    class ConfiguredDrone(Drone):
        # Override the _config class attribute with our new values
        _config = config
    
    return ConfiguredDrone