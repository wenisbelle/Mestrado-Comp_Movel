import enum
import logging
from typing import TypedDict
import numpy as np
from dataclasses import dataclass
import json
from .visualization import MapVisualizer

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
    SHARE_GOTO_POSITION_MESSAGE =2
    RECRUITING_MESSAGE = 3

class HeartBeatMessage(TypedDict):
    message_type: int
    status: int
    sender: int

class ShareMapMessage(TypedDict):
    message_type: int 
    map: list
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
        self.map = np.zeros((100, 2))
        self.threats_found = []
        self.status = DroneStatus.MAPPING

        if Drone.visualizer is None:
            # We have 3 drones in the simulation.
            # I have to think a better way to do this.
            Drone.visualizer = MapVisualizer(num_drones=3)

        self.visualizer.update_map(self.provider.get_id(), self.map)

        command = GotoCoordsMobilityCommand(0, 0, 10)
        self.provider.send_mobility_command(command)
        
        self.provider.schedule_timer("mobility",self.provider.current_time() + 5)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 1)

        configuration = CameraConfiguration(20,30,180,0)
        self.camera = CameraHardware(self, configuration)

    def camera_routine(self):
        detected_nodes = self.camera.take_picture()
        for node in detected_nodes:
            #logging.info(f"Detected point of interest at {node['position']} and type {node['type']}") # saida da camera me da o id do node
            if node['type'] < len(self.map):
                self.map[node['type'], 0] = 1.0
                self.map[node['type'], 1] = self.provider.current_time()

        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map)

    def new_mobility_command(self):
        empty_cells = self.check_empty_cells()                   
        logging.info(f"Node {self.provider.get_id()}. There are {len(empty_cells)} empty cells")

        if empty_cells:
            new_target = float(np.random.choice(empty_cells))
            #logging.info(f"Arrived at destination. Going to cell {new_target}")
            x = (new_target // 10) * 10 - len(self.map)/2
            y = (new_target % 10) * 10 - len(self.map)/2
            self.goto_command = np.array([x, y, 10])
        else:
            #logging.warning(f"No empty cells found. Going to base")
            self.goto_command = (0,0, 0)
        
        command = GotoCoordsMobilityCommand(*self.goto_command)      
        self.provider.send_mobility_command(command)
    
    def send_heartbeat(self):
        #self._log.info(f"Sending heartbeat ...")
        message: HeartBeatMessage = {
            'message_type': MessageType.HEARTBEAT_MESSAGE.value,
            'status': self.status.value,
            'sender': self.provider.get_id()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def compare_maps(self, incoming_map: np.ndarray) -> np.ndarray:
        # Condition: True where the incoming map has a more recent timestamp
        condition = incoming_map[:, 1] > self.map[:, 1]
        # Reshape to apply the condition to both columns of the map
        condition_2d = condition[:, np.newaxis]
        # Where True, take from incoming_map; otherwise, keep the current map's row
        return np.where(condition_2d, incoming_map, self.map)
    
    def check_empty_cells(self):
        return np.where(self.map[:, 0] == 0)[0].tolist()

    def received_heartbeat(self, data: dict):
        heartbeat_msg: HeartBeatMessage = data
        self._log.info(f"Received heartbeat from {heartbeat_msg['sender']}")

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
        #self._log.info(f"Received map from {share_map_msg['sender']}.")
        updated_map = self.compare_maps(np.array(share_map_msg['map']))
        #self._log.info(f"Updated map from {share_map_msg['sender']}.")
        #empty_cells = self.check_empty_cells()
        #self._log.info(f"Node {self.provider.get_id()}. There are {len(empty_cells)} empty cells.")
        #self._log.info(f"Node {self.provider.get_id()}: Map updated from Node {data['sender']}.")
        self.visualizer.update_map(self.provider.get_id(), self.map)

        return updated_map
        
    def handle_timer(self, timer: str) -> None:
        if timer == "camera":
            self.camera_routine()
            self.provider.schedule_timer("camera", self.provider.current_time() + 0.5)
        
        if timer == "mobility": 
            if self.drone_position is not None:
                current_pos_array = np.array(self.drone_position)    
            
            if np.linalg.norm(current_pos_array - self.goto_command) < 1:
                self.new_mobility_command()

            self.provider.schedule_timer(
                "mobility",
                self.provider.current_time() + 5
            )

        if timer == "heartbeat":
            self.send_heartbeat()
            self.provider.schedule_timer("heartbeat", self.provider.current_time() + 1)
            

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

        else:
            self._log.warning(f"Received message with unknown type: {msg_type}")

    def handle_telemetry(self, telemetry: Telemetry) -> None:
        self.drone_position = telemetry.current_position


    def finish(self) -> None:
        logging.info(f"Final map: {self.map}")