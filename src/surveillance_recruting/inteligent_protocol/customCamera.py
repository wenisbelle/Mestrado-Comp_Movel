
import math
from typing import List, Optional, Any

from gradysim.simulator.extension.camera import CameraHardware, DetectedNode



class AdvancedDetectedNode(DetectedNode):
    is_threat: Optional[bool]
    threat_level: Optional[int]


class CustomCamera(CameraHardware):
    """
    A specialized camera that inherits from CameraHardware and overrides
    the take_picture method to add threat detection logic.
    """
    def take_picture(self) -> List[AdvancedDetectedNode]: # Note the updated return type
        """
        This simulated camera hardware is able to detect other nodes within its are of detection. This method returns
        the list of nodes currently inside the area of detection of the camera.
        
        This overridden version also checks for the 'is_threat' attribute.
        Returns:
            A list of detected nodes
        """
        if self._mobility is None:
            return []

        node_position = self._provider.node.position

        other_nodes = [node for node in self._mobility.nodes.values() if node.id != self._provider.node.id]
        
        detected_nodes: List[AdvancedDetectedNode] = [] # Initialize with the new type
        for node in other_nodes:
            other_node_position = node.position
            relative_vector = (
                other_node_position[0] - node_position[0],
                other_node_position[1] - node_position[1],
                other_node_position[2] - node_position[2]
            )

            # Check if the node is within the camera's reach
            distance = math.sqrt(relative_vector[0] ** 2 + relative_vector[1] ** 2 + relative_vector[2] ** 2)
            if distance > self._configuration.camera_reach:
                continue

            if distance > 0:
                # Check if the angle between vectors is less than theta
                normalized_relative_vector = (
                    relative_vector[0] / distance,
                    relative_vector[1] / distance,
                    relative_vector[2] / distance
                )
                dot_product = (
                    self._camera_vector[0] * normalized_relative_vector[0] +
                    self._camera_vector[1] * normalized_relative_vector[1] +
                    self._camera_vector[2] * normalized_relative_vector[2]
                )
                angle = math.acos(dot_product) - 1e-6 # Tolerance
                if angle > self._camera_theta:
                    continue
                        
            protocol_instance = node.protocol_encapsulator.protocol
            threat_value = getattr(protocol_instance, 'is_threat', None)
            threat_level = getattr(protocol_instance, 'threat_level', None)

            detected_nodes.append({
                'position': other_node_position,
                'type': node.id, 
                'is_threat': threat_value, 
                'threat_level': threat_level  
            })
            
        return detected_nodes