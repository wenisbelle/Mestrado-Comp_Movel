import enum
import logging
from typing import TypedDict, Type
import numpy as np
from dataclasses import dataclass
import json
import random
from .visualization import MapVisualizer
from .cluster import ClusterDetector
from .customCamera import CustomCamera
from dataclasses import asdict

from gradysim.protocol.interface import IProtocol
from gradysim.protocol.messages.telemetry import Telemetry
from gradysim.protocol.messages.mobility import GotoCoordsMobilityCommand
from gradysim.simulator.extension.camera import CameraConfiguration
from gradysim.simulator.extension.visualization_controller import VisualizationController
from gradysim.protocol.messages.communication import SendMessageCommand, BroadcastMessageCommand


@dataclass
class Threat:
    level: int
    position_id: int
    timestamp: float 

class DroneStatus(enum.Enum):
    MAPPING = 0
    RECRUITING = 1
    RECRUITED = 2
    ENGAGING = 3
    CREATING_FORMATION = 4

class MessageType(enum.Enum):
    MAPPING_HEARTBEAT_MESSAGE = 0
    SHARE_MAP_MESSAGE = 1
    SHARE_GOTO_POSITION_MESSAGE = 2
    RECRUITING_HEATBEAT_MESSAGE = 3
    CREATE_FORMATION_MESSAGE = 4
    ARRIVED_AT_FORMATION_MESSAGE = 5
    ENGAGEMENT_COMPLETED_MESSAGE = 6
    LIST_OF_THREATS_MESSAGE = 7

class MappingHeartBeatMessage(TypedDict):
    message_type: int
    sender: int

class RecruitingHeartBeatMessage(TypedDict):
    message_type: int
    sender: int
    sender_position: list

class CreateFormationMessage(TypedDict):
    message_type: int
    sender: int
    sender_position: list

class ArrivedAtFormationMessage(TypedDict):
    message_type: int
    sender: int
    has_drone_arrived: bool
    
class ShareMapMessage(TypedDict):
    message_type: int 
    map: list
    sender: int

class SendGoToMessage(TypedDict):
    message_type: int 
    goto: list
    sender: int

class SendEngagementCompletedMessage(TypedDict):
    message_type: int
    sender: int
    point_of_interest_id: int

class SendListOfThreatsMessage(TypedDict):
    message_type: int
    sender: int
    threats: list

class PointOfInterest(IProtocol):
    
    threat_count = 0
    MAX_THREATS = 10
    
    is_threat: int
    threat_level: int
    visualization: VisualizationController
    _log: logging.Logger

    def initialize(self) -> None:
        self._log = logging.getLogger()
        self.is_threat = 0
        self.threat_level = 0
        self.visualization = VisualizationController(self)
        self.green = [0.0, 255.0, 0.0]
        self.red = [255.0, 0.0, 0.0]
       
        self._log.info(f"Point of Interest {self.provider.get_id()} initialized as safe.")
        self.provider.schedule_timer("inital_paint",self.provider.current_time() + 2)

        jitter = random.uniform(0, 100)
        self.provider.schedule_timer("check_threats", self.provider.current_time() + jitter)

    def handle_timer(self, timer: str) -> None:
        if timer == "inital_paint":
            self.visualization.paint_node(self.provider.get_id(), self.green)
        
        if timer == "check_threats":
            if self.is_threat == 1:
                pass
            elif self.is_threat == 0 and PointOfInterest.threat_count < PointOfInterest.MAX_THREATS:
                if random.uniform(0,1) < 0.05: 
                    self.is_threat = 1
                    PointOfInterest.threat_count += 1
                    self._log.info(f"Point of Interest {self.provider.get_id()} has become a THREAT!")
                    self.visualization.paint_node(self.provider.get_id(), self.red)

                    self.threat_level = random.randint(1, 3)
                    self._log.info(f"Point of Interest {self.provider.get_id()} threat level set to {self.threat_level}")
            
            self.provider.schedule_timer("check_threats",self.provider.current_time() + 10)
            

    def handle_packet(self, message: str) -> None:
        data: dict = json.loads(message)

        if 'message_type' not in data:
           self._log.warning(f"Received message without a message_type: {data}")
           return
        
        msg_type = data['message_type']

        if msg_type == MessageType.ENGAGEMENT_COMPLETED_MESSAGE.value:            
            self._log.info(f"Point of Interest {self.provider.get_id()} is not a threat anymore.")
            self.is_threat = 0
            self.visualization.paint_node(self.provider.get_id(), self.green)
            PointOfInterest.threat_count -= 1

            self.provider.schedule_timer("check_threats",self.provider.current_time() + 10)

   
    
    def handle_telemetry(self, telemetry: Telemetry) -> None:
        pass

    def finish(self) -> None:
        pass


class Drone(IProtocol):
    camera: CustomCamera
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

        ## Threats found
        self.threats_found = []
        self.threat_arrival_time = None
        self.SCALE_TO_THREAT_LEVEL = 10
        
        ## Initial state
        self.status = DroneStatus.MAPPING           
        self.drones_in_formation = []
        self.drones_creating_formation = []
        self.drone_leader_id = None

        #self._log.info(f"Drone {self.provider.get_id()} uncertainty rate {self.UNCERTAINTY_RATE}, vanishing time {self.VANISHING_UPDATE_TIME}, map threshold {self.MAP_THRESHOLD}")

        ## Visualization and cluster plugins
        self.cluster_detector = ClusterDetector(threshold=self.MAP_THRESHOLD, min_size=1, map_width=self.MAP_WIDTH, map_height=self.MAP_HEIGHT)
        self.last_drone_interaction_time = np.zeros(self.NUMBER_OF_DRONES)  

        #if Drone.visualizer is None:
        #    # We have 3 drones in the simulation.
        #    # I have to think a better way to do this.
        #    Drone.visualizer = MapVisualizer(num_drones=self.NUMBER_OF_DRONES, map_size=self.MAP_WIDTH * self.MAP_HEIGHT, threshold=self.MAP_THRESHOLD)


        self.goto_command = np.array([random.uniform(-5*self.MAP_WIDTH, 5*self.MAP_WIDTH), random.uniform(-5*self.MAP_HEIGHT, 5*self.MAP_HEIGHT), 10])
        command = GotoCoordsMobilityCommand(*self.goto_command)
        self.provider.send_mobility_command(command)
        
        ## First callbacks
        self.provider.schedule_timer("mapping_mobility",self.provider.current_time() + 1)
        self.provider.schedule_timer("camera",self.provider.current_time() + 1)
        self.provider.schedule_timer("heartbeat",self.provider.current_time() + 1)
        self.vanishing_map_routine()
        self.provider.schedule_timer("vanishing_map", self.provider.current_time() + self.VANISHING_UPDATE_TIME)

        ## Camera Configuration
        configuration = CameraConfiguration(20,30,180,0)
        self.camera = CustomCamera(self, configuration)

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

                #### Check if there are threats
                is_threat = node['is_threat']
                if is_threat:
                    ### First update the drone status if it is mapping
                    if self.status == DroneStatus.MAPPING:
                        self.status = DroneStatus.RECRUITING
                        
                        ### Starting this new logic mobility timer
                        self.provider.schedule_timer("recruiting_mobility", self.provider.current_time())

                        self._log.info(f"Drone {self.provider.get_id()} changing status to RECRUITING due to threat found at position id {node['type']}")

                    ### Then register the threat
                    if node['type'] in [threat.position_id for threat in self.threats_found]:
                        # Just update timestamp
                        for threat in self.threats_found:
                            if threat.position_id == node['type']:
                                threat.timestamp = self.provider.current_time()                                
                    else:                
                        threat_level = node['threat_level']
                        threat = Threat(level=threat_level, position_id=node['type'], timestamp=self.provider.current_time())
                        self.threats_found.append(threat)
                        ### Adding the flated index of the threat to the list of found threats 
                        
                    self._log.info(f"Threats found {self.threats_found}")
                
                                
                # Checking the total uncertainty
                self.total_uncertainty = self.map[:,:,0].sum()
                #self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has total uncertainty of {self.total_uncertainty}")         

        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])

    def vanishing_map_routine(self):
        self.map[:, :, 0] = self.map[:, :, 0] + self.UNCERTAINTY_RATE
        ## Cheching the number of cells abouve the threshold
        discovered_cells = np.sum(self.map[:, :, 0] <= self.MAP_THRESHOLD)
        #self._log.info(f"At time: {self.provider.current_time()}, node {self.provider.get_id()} map has a total of {discovered_cells} cell abouve threshold")

        ## Checking if the cell was visited
        self.is_cell_visited[self.map[:, :, 1] > 0.0] = 1
        #self._log.info(f"At time: {self.provider.current_time()}, the node {self.provider.get_id()} has {self.MAP_WIDTH*self.MAP_HEIGHT - np.sum(self.is_cell_visited)} unvisited cells")
        if self.visualizer:
            self.visualizer.update_map(self.provider.get_id(), self.map[:,:,0])
        

    def internal_mobility_command(self):
        clusters = self.cluster_detector.find_clusters(self.map[:, :, 0])                   
        #logging.info(f"Node {self.provider.get_id()}. Found {len(clusters)} clusters")
        map_center_offset = (self.MAP_WIDTH * 10) / 2

        if len(clusters) > 1:
            cluster_fitness_scores = self.cluster_detector.cluster_fitness(
                self.map[:, :, 0],
                clusters, 
                self.drone_position, 
                distance_norm=self.MAP_WIDTH*self.MAP_HEIGHT, 
                cluster_size_norm=1
            )

            target_coords = self.cluster_detector.choose_one_cluster(cluster_fitness_scores)
            target_row, target_col = target_coords

            #self._log.info(f"Arrived at destination. Going to cell {target_row}, {target_col}")
            
            x_goto = target_row * 10 - map_center_offset
            y_goto = target_col * 10 - map_center_offset            
            self.goto_command = np.array([x_goto, y_goto, 10])
        
        elif len(clusters) == 1:
            maximum_cells_coords = self.check_maximum_cells()                 
            #logging.info(f"Node {self.provider.get_id()}. There are {len(maximum_cells_coords)} minimum cells")

            if maximum_cells_coords:
                target_row, target_col = random.choice(maximum_cells_coords)
                #logging.info(f"Arrived at destination. Going to cell {target_row}, {target_col}")
                x = (target_row) * 10 - map_center_offset
                y = (target_col) * 10 - map_center_offset
                self.goto_command = np.array([x, y, 10])

        else:
            maximum_cells_coords = self.check_maximum_cells()                 
            #logging.info(f"Node {self.provider.get_id()}. There are {len(maximum_cells_coords)} minimum cells")
            if maximum_cells_coords:
                target_row, target_col = random.choice(maximum_cells_coords)
                #logging.info(f"Arrived at destination. Going to cell {target_row}, {target_col}")
                x = (target_row) * 10 - map_center_offset
                y = (target_col) * 10 - map_center_offset
                self.goto_command = np.array([x, y, 10])
        
        command = GotoCoordsMobilityCommand(*self.goto_command)      
        self.provider.send_mobility_command(command)

    def external_mobility_command(self):
        clusters = self.cluster_detector.find_clusters(self.map[:, :, 0])                   
        #logging.info(f"Node {self.provider.get_id()}. Found {len(clusters)} clusters")
        map_center_offset = (self.MAP_WIDTH * 10) / 2
        
        if len(clusters) >= 2:
            cluster_fitness_scores = self.cluster_detector.cluster_fitness(
                self.map[:, :, 0],
                clusters, 
                self.drone_position,  
                distance_norm=self.MAP_WIDTH*self.MAP_HEIGHT, 
                cluster_size_norm=1
            )

            target_coords = self.cluster_detector.choose_two_clusters(cluster_fitness_scores)
            target_row_1, target_col_1 = target_coords[0]
            target_row_2, target_col_2 = target_coords[1]
            
            x_goto = target_row_1 * 10 - map_center_offset
            y_goto = target_col_1 * 10 - map_center_offset            
            self.goto_command = np.array([x_goto, y_goto, 10])

            x_send_command = target_row_2 * 10 - map_center_offset
            y_send_command = target_col_2 * 10 - map_center_offset
            send_command = np.array([x_send_command, y_send_command, 10])
        
        elif len(clusters) < 2 and len(clusters) >= 0:
            maximum_cells_coords = self.check_maximum_cells()                 
            #logging.info(f"Node {self.provider.get_id()}. There are {len(maximum_cells_coords)} maximum cells")

            if len(maximum_cells_coords) >= 2:
                chosen_pairs = random.sample(maximum_cells_coords, 2)
                target_row_1, target_col_1 = chosen_pairs[0]
                target_row_2, target_col_2 = chosen_pairs[1]
                
                x_goto = (target_row_1) * 10 - map_center_offset
                y_goto = (target_col_1) * 10 - map_center_offset
                self.goto_command = np.array([x_goto, y_goto, 10])
                
                x_send_command = (target_row_2) * 10 - map_center_offset
                y_send_command = (target_col_2) * 10 - map_center_offset
                send_command = np.array([x_send_command, y_send_command, 10])

            elif len(maximum_cells_coords)==1:
                target_row, target_col = maximum_cells_coords[0]
                x = (target_row) * 10 - map_center_offset
                y = (target_col) * 10 - map_center_offset
                self.goto_command = np.array([x, y, 10])
                #self._log.info(f"Only one maximum cell found. Sending a random command to the other UAV.")
                send_command = np.array([random.uniform(-5*self.MAP_WIDTH, 5*self.MAP_WIDTH), random.uniform(-5*self.MAP_HEIGHT, 5*self.MAP_HEIGHT), 10])

        else:
            #logging.warning(f"No empty cells found. Going to base")
            self.goto_command = np.array([random.uniform(-5*self.MAP_WIDTH, 5*self.MAP_WIDTH), random.uniform(-5*self.MAP_HEIGHT, 5*self.MAP_HEIGHT), 10])
            send_command = np.array([random.uniform(-5*self.MAP_WIDTH, 5*self.MAP_WIDTH), random.uniform(-5*self.MAP_HEIGHT, 5*self.MAP_HEIGHT), 10])

        return send_command        

    
    def send_mapping_heartbeat(self):
        message: MappingHeartBeatMessage = {
            'message_type': MessageType.MAPPING_HEARTBEAT_MESSAGE.value,
            'sender': self.provider.get_id()
        }
        command = BroadcastMessageCommand(json.dumps(message))
        self.provider.send_communication_command(command)

    def send_recuiting_heartbeat(self):
        message: RecruitingHeartBeatMessage = {
            'message_type': MessageType.RECRUITING_HEATBEAT_MESSAGE.value,
            'sender': self.provider.get_id(),
            'sender_position': np.array(self.drone_position).tolist(), 
        }
        #self._log.info(f"Sending recruiting position {message['sender_position']}")
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

    def mapping_received_heartbeat(self, data: dict):
        heartbeat_msg: MappingHeartBeatMessage = data
        #self._log.info(f"Received heartbeat from {heartbeat_msg['sender']}")
        message: ShareMapMessage = {
            'message_type': MessageType.SHARE_MAP_MESSAGE.value,
            'map': self.map.tolist(),
            'sender': self.provider.get_id()
        }
        destination_id = heartbeat_msg['sender']                
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)

    def sending_create_formation_message(self, destination_id: int):
        ### send to the other drone the formation message
        ### In this case it does not need to send the is_drone_arrived as True
        message: CreateFormationMessage = {
            'message_type': MessageType.CREATE_FORMATION_MESSAGE.value,
            'sender': self.provider.get_id(),
            'sender_position': np.array(self.drone_position).tolist()
        }
        destination_id = destination_id
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)

    def received_recruiting_while_mapping(self, data: dict):
        ### In the future we can decide strategies and negotiations here
        ## Changing status to creating formation
        self.status = DroneStatus.CREATING_FORMATION
        self._log.info(f"Drone {self.provider.get_id()} changing status to CREATING_FORMATION due to heartbeat from drone {data['sender']}")
        
        #### Once it is going to the position, send one last mapping heartbeat to inform the other drone
        self.send_mapping_heartbeat()

        #### Assign the leader drone
        self.drone_leader_id = data['sender']
        
        ### Go to the position sent by the other drone
        heartbeat_msg: RecruitingHeartBeatMessage = data
        self.goto_command = heartbeat_msg['sender_position']
        command = GotoCoordsMobilityCommand(*self.goto_command)
        self.provider.send_mobility_command(command)

        self.provider.schedule_timer(
                    "going_to_formation",
                    self.provider.current_time() + 1)
        
    def send_list_of_threats(self, destination_id: int):
        threats_list_as_dicts = [asdict(t) for t in self.threats_found]

        message: SendListOfThreatsMessage = {
            'message_type': MessageType.LIST_OF_THREATS_MESSAGE.value,
            'sender': self.provider.get_id(),
            'threats': threats_list_as_dicts
            }
        command = SendMessageCommand(json.dumps(message), destination_id)
        self.provider.send_communication_command(command)

        
    def received_recruiting_while_recruiting(self, data:dict):
        self._log.info(f"Drone {self.provider.get_id()} received recruiting heartbeat from drone {data['sender']} while RECRUITING.")
        #################################################################################################
        #################################################################################################
        #### TO DO: get the threats found by the other drone and merge them.
        #################################################################################################
        #################################################################################################
        if self.provider.get_id() > data['sender']:
            ## The drone with the highest id becames the leader.
            self.received_mapping_while_recruiting(data)
            self._log.info(f"Drone {self.provider.get_id()} becoming leader over drone {data['sender']}")
        else:
            self.received_recruiting_while_mapping(data)
            self._log.info(f"Drone {self.provider.get_id()} becoming follower under drone {data['sender']}")

            #### Send the list of threats found to the leader drone
            self.send_list_of_threats(data['sender'])
             



    def received_mapping_while_recruiting(self, data:dict):
        ## Changing status to creating formation
        self.status = DroneStatus.CREATING_FORMATION
        self._log.info(f"Drone {self.provider.get_id()} changing status to CREATING_FORMATION due to message from drone {data['sender']}")

        ### Send one last recruiting heartbeat to inform the other drone
        self.send_recuiting_heartbeat()

        ### Stay at position
        ## Once this status has no timer with mobility command the drone will stay at position
        self.goto_command = self.drone_position
        command = GotoCoordsMobilityCommand(*self.goto_command)
        self.provider.send_mobility_command(command)

        ## Send the creating formation to the other drone
        self.drones_creating_formation.append(data['sender'])
        self.sending_create_formation_message(data['sender'])
        self._log.info(f"Drone {self.provider.get_id()} sending create formation message to drone {data['sender']}")
        ## Schedule a timer to do it reapetedly in case the message is lost, until the other drone confirms arrival
        self.provider.schedule_timer("command_create_formation", self.provider.current_time() + 1.0)

    def send_message_to_point_of_interest(self, point_of_interest_id: int):
        message: SendEngagementCompletedMessage = {
            'message_type': MessageType.ENGAGEMENT_COMPLETED_MESSAGE.value,
            'sender': self.provider.get_id(),
            'point_of_interest_id': point_of_interest_id
        }
        command = SendMessageCommand(json.dumps(message), point_of_interest_id)
        self.provider.send_communication_command(command)
        self._log.info(f"Drone {self.provider.get_id()} sent engagement completed message to point of interest {point_of_interest_id}")

    def send_formation_back_to_mapping(self):
        for drone_id in self.drones_in_formation:
            message: SendEngagementCompletedMessage = {
                'message_type': MessageType.ENGAGEMENT_COMPLETED_MESSAGE.value,
                'sender': self.provider.get_id(),
                'point_of_interest_id': drone_id
            }
            command = SendMessageCommand(json.dumps(message), drone_id)
            self.provider.send_communication_command(command)
            self._log.info(f"Drone {self.provider.get_id()} sent engagement completed message to drone {drone_id} to change back to MAPPING status")
        


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
        
        if timer == "mapping_mobility": 
            if self.drone_position is not None:
                current_pos_array = np.array(self.drone_position)    
            
            if self.status == DroneStatus.MAPPING:
                if np.linalg.norm(current_pos_array - self.goto_command) < 1:
                    self.internal_mobility_command()

                self._log.info(f"Drone {self.provider.get_id()} mapping mobility timer executed.")

                self.provider.schedule_timer(
                    "mapping_mobility",
                    self.provider.current_time() + 1)
                
        if timer == "recruiting_mobility": 
            if self.drone_position is not None:
                current_pos_array = np.array(self.drone_position) 
            
            ### Maybe in the future we can add another strategy for recruiting
            if self.status == DroneStatus.RECRUITING:
                if np.linalg.norm(current_pos_array - self.goto_command) < 1:
                    self.internal_mobility_command()

                self._log.info(f"Drone {self.provider.get_id()} recruiting mobility timer executed.")

                self.provider.schedule_timer(
                    "recruiting_mobility",
                    self.provider.current_time() + 1)
                
        if timer == "command_create_formation":
            if self.status == DroneStatus.CREATING_FORMATION:
                ## Send the creating formation to all other drones that are creating formation with this drone
                for drone_id in self.drones_creating_formation:
                    self.sending_create_formation_message(drone_id)
                    self._log.info(f"Drone {self.provider.get_id()} sending create formation message to drone {drone_id}") 

                self._log.info(f"Drone {self.provider.get_id()} resending create formation messages to drones {self.drones_creating_formation}")               
                ## Schedule a timer to do it reapetedly in case the message is lost, until the other drone confirms arrival
                self.provider.schedule_timer("command_create_formation", self.provider.current_time() + 1.0)

        if timer == "going_to_formation":
            if self.status == DroneStatus.CREATING_FORMATION:
                self._log.info(f"Drone {self.provider.get_id()} going to formation timer executed. Current position {self.drone_position}, going to {self.goto_command}")
                ### If the drone has arrived, it changes its status to RECRUITED and starts:
                ### timer to just follow the leader position
                ### send message to confirm arrival
                current_pos_arr = np.array(self.drone_position)
                goto_cmd_arr = np.array(self.goto_command)
                
                if np.linalg.norm(current_pos_arr - goto_cmd_arr) < 1:
                    self.status = DroneStatus.RECRUITED
                    self._log.info(f"Drone {self.provider.get_id()} changing status to RECRUITED as it has arrived at formation point.")

                    ### Send arrival message
                    message: ArrivedAtFormationMessage = {
                        'message_type': MessageType.ARRIVED_AT_FORMATION_MESSAGE.value,
                        'sender': self.provider.get_id(),
                        'has_drone_arrived': True
                    }
                    destination_id = self.drone_leader_id
                    command = SendMessageCommand(json.dumps(message), destination_id)
                    self.provider.send_communication_command(command)                                

                    ### Start this timer
                    self.provider.schedule_timer(
                        "follow_leader_mobility",
                        self.provider.current_time() + 1.0)  
                
                else:
                    ### Just keep waiting to arrive
                    self.provider.schedule_timer(
                        "going_to_formation",
                        self.provider.current_time() + 1.0)

        if timer == "follow_leader_mobility":
            if self.status == DroneStatus.RECRUITED:
                ## Just follow the leader position
                self._log.info(f"Drone {self.provider.get_id()} is following leader drone {self.drone_leader_id} to position {self.goto_command}")

                self.provider.schedule_timer(
                    "follow_leader_mobility",
                    self.provider.current_time() + 1.0)
                
            if self.status == DroneStatus.ENGAGING or self.status == DroneStatus.RECRUITING:
                ## This is the leader. Send the position to the other drones in formation
                for drone_id in self.drones_in_formation:
                    self.send_goto_command(np.array(self.drone_position), drone_id)
                    self._log.info(f"Drone {self.provider.get_id()} is the leader, sending {self.drone_position} to drone {drone_id}")

                self.provider.schedule_timer(
                    "follow_leader_mobility",
                    self.provider.current_time() + 1.0)
                
        if timer == "engaging_mobility":
            if self.status == DroneStatus.ENGAGING:
                ### Send to the threat position
                
                if not self.threats_found:
                    self._log.info(f"There are no more threats. Changing status to MAPPING.")
                    
                    #### Change the status back to mapping
                    self.status = DroneStatus.MAPPING

                    #### Send message to all other drones in formation to change back to mapping
                    self.send_formation_back_to_mapping()
                    
                    #### Reseting the threat arrival time
                    self.threat_arrival_time = None

                    ### Clean the drones in formation list
                    self.drones_in_formation = []

                    #### reset the drones creating formation list
                    self.drones_creating_formation = []

                    #### reset leader id
                    self.drone_leader_id = None


                    ### Get to Drone back to mapping activity
                    self.provider.schedule_timer(
                    "mapping_mobility",
                    self.provider.current_time() + 1)

                else:
                    ### For now there is just one threat, but in the future we can 
                    ### apply a better logic to choose which threat to engage first
                    threat_position = self.threats_found[0].position_id - self.NUMBER_OF_DRONES
                    map_center_offset = (self.MAP_WIDTH * 10) / 2
                    target_row = threat_position // self.MAP_WIDTH
                    target_col = threat_position % self.MAP_WIDTH

                    x_goto = target_row * 10 - map_center_offset
                    y_goto = target_col * 10 - map_center_offset            

                    self.goto_command = np.array([x_goto, y_goto, 10])
                    command = GotoCoordsMobilityCommand(*self.goto_command)
                    self.provider.send_mobility_command(command)

                    self._log.info(f"Drone {self.provider.get_id()} going to threat location, which is at cell {target_row}, {target_col}, coordinates {self.goto_command}")

                    if np.linalg.norm(np.array(self.drone_position) - self.goto_command) < 1:
                        
                        if self.threat_arrival_time is None:
                            self.threat_arrival_time = self.provider.current_time()
                            self._log.info(f"Drone {self.provider.get_id()} has arrived at threat location. Starting engagement timer.")
                        
                        elif (self.provider.current_time() - self.threat_arrival_time) < self.SCALE_TO_THREAT_LEVEL * self.threats_found[0].level:
                            time_remaining = 10 - (self.provider.current_time() - self.threat_arrival_time)
                            self._log.info(f"Neutralizing threat... {time_remaining:.1f}s remaining.")

                        else:
                            self._log.info(f"MISSION ACCOMPLISHED!")
                            self._log.info(f"Removing threat at position id {self.threats_found[0].position_id} from the list.")
                            
                            ### Change the point back to safe
                            self.send_message_to_point_of_interest(self.threats_found[0].position_id)
                            
                            ### Remove the threat from the list
                            self.threats_found.pop(0)

                            ### Restart this timer to go to next threat or back to mapping
                            self.threat_arrival_time = None


                    self.provider.schedule_timer(
                        "engaging_mobility",
                        self.provider.current_time() + 5)

        if timer == "heartbeat":
            if self.status == DroneStatus.MAPPING: 
                self.send_mapping_heartbeat()
                self._log.info(f"Drone {self.provider.get_id()} sent mapping heartbeat.")
            
            if self.status == DroneStatus.RECRUITING:
                self.send_recuiting_heartbeat()
                self._log.info(f"Drone {self.provider.get_id()} sent recruiting heartbeat.")

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

        if msg_type == MessageType.MAPPING_HEARTBEAT_MESSAGE.value:            
            if self.status == DroneStatus.MAPPING:
                self.mapping_received_heartbeat(data)

            elif self.status == DroneStatus.RECRUITING:                
                self.received_mapping_while_recruiting(data)

        elif msg_type == MessageType.RECRUITING_HEATBEAT_MESSAGE.value:
            if self.status == DroneStatus.MAPPING:
                self.received_recruiting_while_mapping(data)

            if self.status == DroneStatus.RECRUITING:
                self.received_recruiting_while_recruiting(data)
        

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


        elif msg_type == MessageType.CREATE_FORMATION_MESSAGE.value:
            formation_msg: CreateFormationMessage = data
            self._log.info(f"Drone {self.provider.get_id()} received create formation message from drone {formation_msg['sender']}. Going to position {formation_msg['sender_position']}    ")
            ## Go to the formation position
            self.goto_command = formation_msg['sender_position'] 
            command = GotoCoordsMobilityCommand(*self.goto_command)
            self.provider.send_mobility_command(command)

            ### Define the sender drone as the new leader
            if self.drone_leader_id == None:
                self.drone_leader_id = formation_msg['sender']          
                      
        
        elif msg_type == MessageType.ARRIVED_AT_FORMATION_MESSAGE.value:
            arrived_msg: ArrivedAtFormationMessage = data
            if arrived_msg['has_drone_arrived']:
                ### Change the variable drones_creating_formation to exclude this drone
                if arrived_msg['sender'] in self.drones_creating_formation:
                    self.drones_creating_formation.remove(arrived_msg['sender'])
                else:
                    self._log.warning(f"Drone {self.provider.get_id()} received arrival confirmation from drone {arrived_msg['sender']} but this drone was not in the creating formation list.")
                
                ### Update the variable drones_in_formation
                self.drones_in_formation.append(arrived_msg['sender'])

                ### This will depend on the trheats found and the number of drones in formation
                self.status = DroneStatus.ENGAGING
                self._log.info(f"Drone {self.provider.get_id()} changing status back to RECRUITING for now as the other drone has arrived at formation point.")
                self.provider.schedule_timer("engaging_mobility", self.provider.current_time() + 1.0)

                self.provider.schedule_timer(
                    "follow_leader_mobility",
                    self.provider.current_time() + 1.0)  
        
        elif msg_type == MessageType.ENGAGEMENT_COMPLETED_MESSAGE.value:
            if self.status == DroneStatus.RECRUITED:
                ### Let's check if the sender is the leader
                if data['sender'] == self.drone_leader_id:
                    ### Change status back to MAPPING
                    self.status = DroneStatus.MAPPING
                    self._log.info(f"Drone {self.provider.get_id()} changing status back to MAPPING")

                    #### Empty the formation lists
                    self.drones_in_formation = []
                    self.drones_creating_formation = []

                    ### Reset the threats found
                    self.threats_found = []
                    
                    ### Reset leader id
                    self.drone_leader_id = None

                    ### Get back to mapping mobility
                    self.provider.schedule_timer(
                        "mapping_mobility",
                        self.provider.current_time() + 1)
        
        elif msg_type == MessageType.LIST_OF_THREATS_MESSAGE.value:
            received_threats_data = data['threats']

            new_threats = []
            for t_dict in received_threats_data:
                # Recreate the Threat object from the dictionary
                threat_obj = Threat(
                    level=t_dict['level'], 
                    position_id=t_dict['position_id'], 
                    timestamp=t_dict['timestamp']
                )
                new_threats.append(threat_obj)

            for original_threat in self.threats_found:
                for threat in new_threats:
                    if threat.position_id == original_threat.position_id:
                        if threat.timestamp > original_threat.timestamp:
                            self._log.info(f"Drone {self.provider.get_id()} updating threat at position id {threat.position_id} with level {threat.level} from drone {data['sender']}")
                            original_threat.level = threat.level
                            original_threat.timestamp = threat.timestamp
                    else:
                        self.threats_found.append(threat)
                        self._log.info(f"Drone {self.provider.get_id()} added new threat from drone {data['sender']} at position id {threat.position_id} with level {threat.level}")    
        
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