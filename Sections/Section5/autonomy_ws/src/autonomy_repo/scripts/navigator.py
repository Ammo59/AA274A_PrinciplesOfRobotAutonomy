#!/usr/bin/env python3

from asl_tb3_lib.navigation import BaseNavigator
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_lib.tf_utils import quaternion_to_yaw
import rclpy                    # ROS2 client library
from rclpy.node import Node     # ROS2 node baseclass
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
import numpy as np
from numpy import linalg
from asl_tb3_lib.navigation import TrajectoryPlan
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import typing as T
from asl_tb3_lib.grids import StochOccupancyGrid2D

class AStar(object):
    """Represents a motion planning problem to be solved using A*"""

    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        self.statespace_lo = statespace_lo         # state space lower bound (e.g., [-5, -5])
        self.statespace_hi = statespace_hi         # state space upper bound (e.g., [5, 5])
        self.occupancy = occupancy                 # occupancy grid (a DetOccupancyGrid2D object)
        self.resolution = resolution               # resolution of the discretization of state space (cell/m)
        self.x_offset = x_init                     
        self.x_init = self.snap_to_grid(x_init)    # initial state
        self.x_goal = self.snap_to_grid(x_goal)    # goal state

        self.closed_set = set()    # the set containing the states that have been visited
        self.open_set = set()      # the set containing the states that are condidate for future expension

        self.est_cost_through = {}  # dictionary of the estimated cost from start to goal passing through state (often called f score)
        self.cost_to_arrive = {}    # dictionary of the cost-to-arrive at state from start (often called g score)
        self.came_from = {}         # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.add(self.x_init)
        self.cost_to_arrive[self.x_init] = 0
        self.est_cost_through[self.x_init] = self.distance(self.x_init,self.x_goal)

        self.path = None        # the final path as a list of states

    def is_free(self, x):
        """
        Checks if a give state x is free, meaning it is inside the bounds of the map and
        is not inside any obstacle.
        Inputs:
            x: state tuple
        Output:
            Boolean True/False
        Hint: self.occupancy is a DetOccupancyGrid2D object, take a look at its methods for what might be
              useful here
        """
        ########## Code starts here ##########

        # Check if implemented
        #raise NotImplementedError("is_free not implemented")

        # Check if x is free using DetOccupancyGrid2D method is_free()
        if (self.occupancy.is_free(np.array(x))) and (self.statespace_hi[0]>x[0]>self.statespace_lo[0]) and (self.statespace_hi[1]>x[1]>self.statespace_lo[1]):
            return True
        return False
        
        ########## Code ends here ##########

    def distance(self, x1, x2):
        """
        Computes the Euclidean distance between two states.
        Inputs:
            x1: First state tuple
            x2: Second state tuple
        Output:
            Float Euclidean distance

        HINT: This should take one line. Tuples can be converted to numpy arrays using np.array().
        """
        ########## Code starts here ##########

        # Check if implemented
        #raise NotImplementedError("distance not implemented")
        
        # Find Euclidean distance using norm function
        return np.linalg.norm(np.array(x1) - np.array(x2))
        
        ########## Code ends here ##########

    def snap_to_grid(self, x):
        """ Returns the closest point on a discrete state grid
        Input:
            x: tuple state
        Output:
            A tuple that represents the closest point to x on the discrete state grid
        """
        return (
            self.resolution * round((x[0] - self.x_offset[0]) / self.resolution) + self.x_offset[0],
            self.resolution * round((x[1] - self.x_offset[1]) / self.resolution) + self.x_offset[1],
        )

    def get_neighbors(self, x):
        """
        Gets the FREE neighbor states of a given state x. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals by an
        amount equal to self.resolution.
        Input:
            x: tuple state
        Ouput:
            List of neighbors that are free, as a list of TUPLES

        HINTS: Use self.is_free to check whether a given state is indeed free.
               Use self.snap_to_grid (see above) to ensure that the neighbors
               you compute are actually on the discrete grid, i.e., if you were
               to compute neighbors by adding/subtracting self.resolution from x,
               numerical errors could creep in over the course of many additions
               and cause grid point equality checks to fail. To remedy this, you
               should make sure that every neighbor is snapped to the grid as it
               is computed.
        """
        neighbors = []
        ########## Code starts here ##########

        # Check if implemented
        #raise NotImplementedError("get_neighbors not implemented")

        # Created grid of possible locations for neighbors
        neighbor_grid = [(0,self.resolution),(self.resolution,self.resolution),
                         (self.resolution,0),(self.resolution,-self.resolution),
                         (0,-self.resolution),(-self.resolution,-self.resolution),
                         (-self.resolution,0),(-self.resolution,self.resolution)]

        # Check if neighbors are possible. If they are, add neighbor tuple to neighbors
        for i in range(0,8):
            x_neighbor = tuple(map(lambda m, n: m + n, x,neighbor_grid[i]))
            x_neighbor = self.snap_to_grid(x_neighbor)
            if(self.is_free(x_neighbor)):
                if x_neighbor in neighbors:
                    continue
                else:
                    neighbors.append(x_neighbor)
        
        ########## Code ends here ##########
        return neighbors

    def find_best_est_cost_through(self):
        """
        Gets the state in open_set that has the lowest est_cost_through
        Output: A tuple, the state found in open_set that has the lowest est_cost_through
        """
        return min(self.open_set, key=lambda x: self.est_cost_through[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location to
        the goal location
        Output:
            A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        Input:
            None
        Output:
            Boolean, True if a solution from x_init to x_goal was found

        HINTS:  We're representing the open and closed sets using python's built-in
                set() class. This allows easily adding and removing items using
                .add(item) and .remove(item) respectively, as well as checking for
                set membership efficiently using the syntax "if item in set".
        """
        ########## Code starts here ##########

        # Check if implemented
        #raise NotImplementedError("solve not implemented")

        # 
        while (len(self.open_set) > 0):
            x_current = self.find_best_est_cost_through()
            if x_current == self.x_goal:
                self.path = self.reconstruct_path()
                return True
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current,x_neigh)
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                elif tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]:
                    continue
                self.came_from[x_neigh] = x_current
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh,self.x_goal)

        return False
        
        ########## Code ends here ##########


class NavNode(BaseNavigator):
    def __init__(self) -> None:
        # give it a default node name
        super().__init__("nav_node")
        self.kp = 2.0
        self.V_PREV_THRES = 0.0001
        self.kpx = 2
        self.kpy = 2
        self.kdx = 2
        self.kdy = 2
        self.t_prev = 0.
        self.v_desired = 0.15
        self.spline_alpha = 0.05

    def compute_heading_control(self,
        state: TurtleBotState,
        goal: TurtleBotState
    ) -> TurtleBotControl:
        head_err = wrap_angle(goal.theta - state.theta)
        om = self.kp * head_err
        new_control = TurtleBotControl()
        new_control.omega = om
        new_control.v = 0.
        return new_control
    
    def compute_trajectory_tracking_control(self,
        state: TurtleBotState,
        plan: TrajectoryPlan,
        t: float,
    ) -> TurtleBotControl:

        dt = t - self.t_prev
        x_d = interpolate.splev(state.x,plan.path_x_spline,der=0)
        xd_d = interpolate.splev(state.x,plan.path_x_spline,der=1)
        xdd_d = interpolate.splev(state.x,plan.path_x_spline,der=2)
        y_d = interpolate.splev(state.y,plan.path_y_spline,der=0)
        yd_d = interpolate.splev(state.y,plan.path_y_spline,der=1)
        ydd_d = interpolate.splev(state.y,plan.path_y_spline,der=2)

        ########## Code starts here ##########
        u1 = xdd_d + self.kpx * (x_d - state.x) + self.kdx * (xd_d - self.V_prev * np.cos(state.theta))
        u2 = ydd_d + self.kpy * (y_d - state.y) + self.kdy * (yd_d - self.V_prev * np.sin(state.theta))

        V = self.V_prev + dt * (u1 * np.cos(state.theta) + u2 * np.sin(state.theta))
        if (V < self.V_PREV_THRES):
            V = self.V_PREV_THRES
        om = (u2 * np.cos(state.theta) - u1 * np.sin(state.theta)) / V
        
        ########## Code ends here ##########

        # save the commands that were applied and the time
        self.t_prev = t
        self.V_prev = V
        self.om_prev = om
        
        control = TurtleBotControl()
        control.v = V
        control.omega = om

        return control
    
    def compute_trajectory_plan(self,
        state: TurtleBotState,
        goal: TurtleBotState,
        occupancy: StochOccupancyGrid2D,
        resolution: float,
        horizon: float,
    ) -> TrajectoryPlan:

        x_init = (state.x, state.y)
        x_goal = (goal.x, goal.y)

        astar = AStar((-horizon, -horizon), (horizon,horizon), x_init, x_goal, occupancy, resolution=resolution)
        if (not astar.solve()):
            return None
        
        path = np.asarray(astar.path)

        if (np.shape(path)[0] < 4):
            return None

        self.V_prev = 0.
        self.om_prev = 0.
        self.t_prev = 0.
        
        ts = np.zeros(len(path))
        path_x = np.array([])
        path_y = np.array([])
        cumulative = 0
        
        # Separate path into x and y components
        for i in range(0, len(path)):
            path_x = np.append(path_x,path[i][0])
            path_y = np.append(path_y,path[i][1])
        
        # Calculate cumulative time for each waypoint
        for i in range(0,len(path)-1):
            ts[i+1] = np.linalg.norm(path[i+1] - path[i]) / self.v_desired + cumulative
            cumulative = ts[i+1]
        
        # Fit cubic splines for x and y
        path_x_spline = interpolate.splrep(ts, path_x, k=3, s=self.spline_alpha)
        path_y_spline = interpolate.splrep(ts, path_y, k=3, s=self.spline_alpha)

        ###### YOUR CODE END HERE ######
        
        traj = TrajectoryPlan(
            path=path,
            path_x_spline=path_x_spline,
            path_y_spline=path_y_spline,
            duration=ts[-1],
        )

        return traj


if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    node = NavNode()    # create the node instance
    rclpy.spin(node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits
