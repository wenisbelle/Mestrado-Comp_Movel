import enum
import logging
from typing import TypedDict
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
        logging.info(f"The cell is {self.is_threat}")

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

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.drone_position = None
        self.goto_command = (0.0, 0.0, 10.0)
        self.MAP_WIDTH = 10
        self.MAP_HEIGHT = 10
        self.map = np.zeros((self.MAP_HEIGHT, self.MAP_WIDTH, 2))
        self.threats_found = []
        self.status = DroneStatus.MAPPING
        self.DECAY_RATE = 0.05
        self.VANISHING_UPDATE_TIME = 10.0
        self.cluster_detector = ClusterDetector(threshold=0.5, min_size=3)
        self.last_drone_interaction_time = np.zeros(2)  # Assuming 2 drones, starting with ID 100

        if Drone.visualizer is None:
            # We have 3 drones in the simulation.
            # I have to think a better way to do this.
            Drone.visualizer = MapVisualizer(num_drones=3)

        self.visualizer.update_map(self.provider.get_id(), self.map.reshape(-1, 2))

        self.goto_command = np.array([random.uniform(-50, 50), random.uniform(-50, 50), 10])
        command = GotoCoordsMobilityCommand(*self.goto_command)
        self.provider.send_mobility_command(command)
        
        self.provider.schedule_timer("mobility",self.provider.current_time() + 5)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 1)
        self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)

        configuration = CameraConfiguration(20,30,180,0)
        self.camera = CameraHardware(self, configuration)

    def camera_routine(self):
        detected_nodes = self.camera.take_picture()
        for node in detected_nodes:
            #logging.info(f"Detected point of interest at {node['position']} and type {node['type']}") # saida da camera me da o id do node
            flat_index = node['type']
            if flat_index < self.MAP_WIDTH * self.MAP_HEIGHT:
                row = flat_index // self.MAP_WIDTH
                col = flat_index % self.MAP_WIDTH

                self.map[row, col, 0] = 1.0 
                self.map[row, col, 1] = self.provider.current_time()

        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map.reshape(-1, 2))

    def vanishing_map_routine(self):
        self.map[:, :, 0] = np.clip(self.map[:, :, 0] - self.DECAY_RATE, 0, 1)
        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map.reshape(-1, 2))
        

    def internal_mobility_command(self):
        clusters = self.cluster_detector.find_zero_clusters(self.map[:, :, 0])                   
        logging.info(f"Node {self.provider.get_id()}. Found {len(clusters)} clusters")

        if len(clusters) >= 3:
            cluster_fitness_scores = self.cluster_detector.cluster_fitness(
                clusters, 
                self.drone_position, 
                self.MAP_WIDTH, 
                self.MAP_HEIGHT, 
                distance_norm=100, 
                cluster_size_norm=5
            )

            target_coords = self.cluster_detector.choose_one_cluster(cluster_fitness_scores)
            target_row, target_col = target_coords

            self._log.info(f"Arrived at destination. Going to cell {target_row}, {target_col}")
            map_center_offset = (self.MAP_WIDTH * 10) / 2
            x_goto = target_row * 10 - map_center_offset
            y_goto = target_col * 10 - map_center_offset            
            self.goto_command = np.array([x_goto, y_goto, 10])
        
        elif len(clusters) < 3 and len(clusters) > 0:
            minimum_cells_coords = self.check_minimum_cells()                 
            logging.info(f"Node {self.provider.get_id()}. There are {len(minimum_cells_coords)} minimum cells")

            if minimum_cells_coords:
                target_row, target_col = random.choice(minimum_cells_coords)
                logging.info(f"Arrived at destination. Going to cell {target_row}, {target_col}")
                x = (target_row) * 10 - self.MAP_WIDTH * 10/2
                y = (target_col) * 10 - self.MAP_HEIGHT * 10/2
                self.goto_command = np.array([x, y, 10])

        else:
            #logging.warning(f"No empty cells found. Going to base")
            self.goto_command = (0,0, 0)
        
        command = GotoCoordsMobilityCommand(*self.goto_command)      
        self.provider.send_mobility_command(command)

    def external_mobility_command(self):
        clusters = self.cluster_detector.find_zero_clusters(self.map[:, :, 0])                   
        logging.info(f"Node {self.provider.get_id()}. Found {len(clusters)} clusters")

        if len(clusters) >= 3:
            cluster_fitness_scores = self.cluster_detector.cluster_fitness(
                clusters, 
                self.drone_position, 
                self.MAP_WIDTH, 
                self.MAP_HEIGHT, 
                distance_norm=100, 
                cluster_size_norm=5
            )

            target_coords = self.cluster_detector.choose_two_clusters(cluster_fitness_scores)
            target_row_1, target_col_1 = target_coords[0]
            target_row_2, target_col_2 = target_coords[1]

            map_center_offset = (self.MAP_WIDTH * 10) / 2
            x_goto = target_row_1 * 10 - map_center_offset
            y_goto = target_col_1 * 10 - map_center_offset            
            self.goto_command = np.array([x_goto, y_goto, 10])

            x_send_command = target_row_2 * 10 - map_center_offset
            y_send_command = target_col_2 * 10 - map_center_offset
            send_command = np.array([x_send_command, y_send_command, 10])
        
        elif len(clusters) < 3 and len(clusters) > 0:
            minimum_cells_coords = self.check_minimum_cells()                 
            logging.info(f"Node {self.provider.get_id()}. There are {len(minimum_cells_coords)} minimum cells")

            if len(minimum_cells_coords)>=2:
                chosen_pairs = random.sample(minimum_cells_coords, 2)
                target_row_1, target_col_1 = chosen_pairs[0]
                target_row_2, target_col_2 = chosen_pairs[1]
                
                x_goto = (target_row_1) * 10 - self.MAP_WIDTH * 10/2
                y_goto = (target_col_1) * 10 - self.MAP_HEIGHT * 10/2
                self.goto_command = np.array([x_goto, y_goto, 10])
                
                x_send_command = (target_row_2) * 10 - self.MAP_WIDTH * 10/2
                y_send_command = (target_col_2) * 10 - self.MAP_HEIGHT
                send_command = np.array([x_send_command, y_send_command, 10])

            elif len(minimum_cells_coords)==1:
                target_row, target_col = minimum_cells_coords[0]
                x = (target_row) * 10 - self.MAP_WIDTH * 10/2
                y = (target_col) * 10 - self.MAP_HEIGHT * 10/2
                self.goto_command = np.array([x, y, 10])
                self._log.info(f"Only one minimum cell found. Sending a random command to the other UAV.")
                send_command = np.array([random.uniform(-50, 50), random.uniform(-50, 50), 10])

        else:
            #logging.warning(f"No empty cells found. Going to base")
            self.goto_command = (0,0, 0)
            send_command = (0,0,0)

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
    
    def check_minimum_cells(self):
        #self._log.info(f"Minimum cell value is {self.map[:,0].min()}")
        min_knowledge = self.map[:, :, 0].min()
        rows, cols = np.where(self.map[:, :, 0] == min_knowledge)
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
        self.visualizer.update_map(self.provider.get_id(), self.map.reshape(-1, 2))

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

            if self.provider.current_time() - self.last_drone_interaction_time[data['sender']-100]  > 3.0: # the drone id starts at 100
                if self.provider.get_id() >= data['sender']:
                    self._log.info(f"Node {self.provider.get_id()} is calculating the new destinations")
                    send_command = self.external_mobility_command()
                    
                    command = GotoCoordsMobilityCommand(*self.goto_command)      
                    self.provider.send_mobility_command(command)

                    self.send_goto_command(send_command, data['sender'])
                self.last_drone_interaction_time[data['sender']-100] = self.provider.current_time() # the drone id starts at 100
        
        elif msg_type == MessageType.SHARE_GOTO_POSITION_MESSAGE.value:
            goto_msg: SendGoToMessage = data
            self._log.info(f"Received goto command from {goto_msg['sender']}. Going to {goto_msg['goto']}")
            self.goto_command = goto_msg['goto']
            command = GotoCoordsMobilityCommand(*self.goto_command)      
            self.provider.send_mobility_command(command)       
        
        else:
            self._log.warning(f"Received message with unknown type: {msg_type}")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.drone_position = telemetry.current_position


    def finish(self) -> None:
        logging.info(f"Final map: {self.map}")