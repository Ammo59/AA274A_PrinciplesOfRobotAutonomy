#!/usr/bin/env python3

import numpy as np
import rclpy

from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    
    def __init__ (self) -> None:
        """ - Call parents init method
            - k_p = 2.0 : proportional control gain 
        """
        super().__init__()
        self.k_p = 2.0
        
        
    def compute_control_with_goal(self, current_state: TurtleBotState, desired_state: TurtleBotState) -> TurtleBotControl:
        """ - Takes in current state and desired state of TurtleBot.
            - They are of type TurtleBotState.
            - Returns a control message of type TurtleBotControl
        """
        
        # calculate heading error [-pi, pi] as the wrapped difference between the goal's theta and state's theta
        heading_err = wrap_angle(desired_state.theta - current_state.theta)
        
        # use the proportional control formula, omega = k_p * err, to compute angular velocity required to correct heading error
        new_omega = self.k_p * heading_err
        
        # create a new TurtleBotControl message and set its omega attribute to be computed angular velocity and return it
        prop_control = TurtleBotControl()
        prop_control.omega = new_omega
        
        return prop_control
    

if __name__ == "__main__":
    rclpy.init() # initialize ROS2 system
    heading_controller = HeadingController() # create an instance (node) of the Heading Controller Class
    rclpy.spin(heading_controller) # spin the node to keep it running and listening for messages
    rclpy.shutdown() # shutdown after spinning
    
    
     
        
        
        