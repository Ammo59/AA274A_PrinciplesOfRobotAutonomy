#!/usr/bin/env python3

import numpy as np
import math
import rclpy
import scipy
# from rclpy.node import Node

from asl_tb3_lib.navigation import BaseNavigator, TrajectoryPlan
from asl_tb3_lib.math_utils import wrap_angle, distance_linear, distance_angular
from asl_tb3_lib.tf_utils import quaternion_to_yaw
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState
from scipy.interpolate import splev, splrep
#from asl_tb3_lib.grids import StochOccupancyGrid2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# from utils import plot_line_segments

# My own imports
# import A_star algo and copy class + methods
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
        
        # check if tuple coordinates are within grid boundaries
        for coordinate in range(len(x)):
            if (x[coordinate] < self.statespace_lo[coordinate]) or (x[coordinate] > self.statespace_hi[coordinate]): # out of bounds
                return False 
        
        # if coordinates are not out of bounds, check if they are colliding with an obstacle
        return self.occupancy.is_free(x)
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
        return np.linalg.norm(np.array(x1) - np.array(x2)) #since we want to be fancy and do it in one line
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
        # check if current state is free
        if self.is_free(x):
            
            # Generate list of potential neighbors
            left = (x[0] - self.resolution, x[1])
            right = (x[0] + self.resolution, x[1])
            up = (x[0], x[1] + self.resolution)
            down = (x[0], x[1] - self.resolution)
            
            diag_up_left = (x[0] - self.resolution, x[1] + self.resolution)
            diag_up_right = (x[0] + self.resolution, x[1] + self.resolution)
            diag_down_right = (x[0] + self.resolution, x[1] - self.resolution)
            diag_down_left = (x[0] - self.resolution, x[1] - self.resolution)
            
            potential_neighbors = [left, right, up, down, diag_up_left, diag_up_right, diag_down_right, diag_down_left]
             
            for n in potential_neighbors:
                snapped_n = self.snap_to_grid(n) # snap neighbors to grid in case they are not, must use new var since tuples are immutable
                if self.is_free(snapped_n):
                    neighbors.append(snapped_n) # repopulate list with new, snapped tuples
        
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

    def plot_path(self, fig_num=0, show_init_label=True):
        """Plots the path found in self.path and the obstacles"""
        if not self.path:
            return

        self.occupancy.plot(fig_num)

        solution_path = np.asarray(self.path)
        plt.plot(solution_path[:,0],solution_path[:,1], color="green", linewidth=2, label="A* solution path", zorder=10)
        plt.scatter([self.x_init[0], self.x_goal[0]], [self.x_init[1], self.x_goal[1]], color="green", s=30, zorder=10)
        if show_init_label:
            plt.annotate(r"$x_{init}$", np.array(self.x_init) + np.array([.2, .2]), fontsize=16)
        plt.annotate(r"$x_{goal}$", np.array(self.x_goal) + np.array([.2, .2]), fontsize=16)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)

        plt.axis([0, self.occupancy.width, 0, self.occupancy.height])

    def plot_tree(self, point_size=15):
        #plot_line_segments([(x, self.came_from[x]) for x in self.open_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        #plot_line_segments([(x, self.came_from[x]) for x in self.closed_set if x != self.x_init], linewidth=1, color="blue", alpha=0.2)
        px = [x[0] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        py = [x[1] for x in self.open_set | self.closed_set if x != self.x_init and x != self.x_goal]
        plt.scatter(px, py, color="blue", s=point_size, zorder=10, alpha=0.2)

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
        while (len(self.open_set) > 0):
            x_current = self.find_best_est_cost_through() 
# find the lowest estimated cost and assign it to current state (x)
            
            # Check if we've reached our goal state
            if x_current == self.x_goal:
                self.path = self.reconstruct_path() #unsure if this is correct, must test later
                return True
            
            # Add and remove x_current from closed and open sets, respectively
            self.open_set.remove(x_current)
            self.closed_set.add(x_current)
            
            # Continue a_star algorithm with searching nearest neighbors
            for x_neigh in self.get_neighbors(x_current):
                if x_neigh in self.closed_set:
                    continue
                
                tentative_cost_to_arrive = self.cost_to_arrive[x_current] + self.distance(x_current, x_neigh) #unsure if I did this correctly, check later when testing
                
                # add neighbor if not in open set already
                if x_neigh not in self.open_set:
                    self.open_set.add(x_neigh)
                
                elif (tentative_cost_to_arrive > self.cost_to_arrive[x_neigh]): #the algo says gt but it may be ge
                    continue
                    
                self.came_from[x_neigh] = x_current # unsure if I did this right, check when testing; syntax corrected
                self.cost_to_arrive[x_neigh] = tentative_cost_to_arrive
                self.est_cost_through[x_neigh] = tentative_cost_to_arrive + self.distance(x_neigh, self.x_goal)
            
        return False
        ########## Code ends here ##########

class DetOccupancyGrid2D(object):
    """
    A 2D state space grid with a set of rectangular obstacles. The grid is
    fully deterministic
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        """Verifies that point is not inside any obstacles by some margin"""
        for obs in self.obstacles:
            if x[0] >= obs[0][0] - self.width * .01 and \
               x[0] <= obs[1][0] + self.width * .01 and \
               x[1] >= obs[0][1] - self.height * .01 and \
               x[1] <= obs[1][1] + self.height * .01:
                return False
        return True

    def plot(self, fig_num=0):
        """Plots the space and its obstacles"""
        fig = plt.figure(fig_num)
        ax = fig.add_subplot(111, aspect='equal')
        for obs in self.obstacles:
            ax.add_patch(
            patches.Rectangle(
            obs[0],
            obs[1][0]-obs[0][0],
            obs[1][1]-obs[0][1],))
        ax.set(xlim=(0,self.width), ylim=(0,self.height))

        
class NavigationNode(BaseNavigator):
    
    def __init__(self) -> None:
        super().__init__("navigation_node")
        
        # gain initialization
        self.k_p = 2.0 # maybe we have a kpx, kpy, kptheta, etc.?
        # self.k_i = 1 # may be unneeded
        self.k_d = 5 # same here as above 2 comments
        
        # do we need self.kpx, kpy kdx kdy? no right?
        
        # velocity initialization
        self.V_desired = 0.15
        self.V_PREV_TRESH = 0.01
        self.V_prev = self.V_PREV_TRESH
        
        # time tracking variable initialization
        self.prev_heading_err = 0.0
        self.t_prev = 0.0
        
        # set spline paramters
        self.spline_alpha = 0.05
        
    def compute_heading_control(self, current_state: TurtleBotState, desired_state: TurtleBotState) -> TurtleBotControl:
        """ - Takes in current state and desired state of TurtleBot.
            - They are of type TurtleBotState.
            - Returns a control message of type TurtleBotControl
        """
        
        # calculate heading error [-pi, pi] as the wrapped difference between the goal's theta and state's theta
        heading_err = wrap_angle(desired_state.theta - current_state.theta)
        
        # use the proportional control formula, omega = k_p * err, to compute angular velocity required to correct heading error
        new_omega = self.k_p * heading_err # + self.k_d * (velocity_err - velocity_err)
        self.prev_heading_err = heading_err
        
        # create a new TurtleBotControl message and set its omega attribute to be computed angular velocity and return it
        prop_control = TurtleBotControl() # this will create a TurtleBotControl message
        prop_control.omega = new_omega
        
        return prop_control
    
    def compute_trajectory_tracking_control(self, state: TurtleBotState, plan: TrajectoryPlan, t:float) -> TurtleBotControl:
        """Overriden from super: Compute control traget using a trajectory tracking controller
           with spline interpolation

        Args:
            state (TurtleBotState): current robot state
            plan (TrajectoryPlan): planned trajetory
            t (float): current timestep

        Returns:
            TurtleBotControl: control command
        """            
        
        dt = t - self.t_prev
        
        # Which one do I use?
        # desired_state = plan.desired_state(t) # x_d, xd_d, xdd_d, y_d, yd_d, ydd_d
        x_d = float(splev(state.x, plan.path_x_spline, der = 0))
        xd_d = float(splev(state.x, plan.path_x_spline, der = 1))
        xdd_d = float(splev(state.x, plan.path_x_spline, der = 2))
        
        y_d = float(splev(state.y, plan.path_y_spline, der = 0))
        yd_d = float(splev(state.y, plan.path_y_spline, der = 1))
        ydd_d = float(splev(state.y, plan.path_y_spline, der = 2))
        
        # if check for potential singularity
        if (self.V_prev < self.V_PREV_THRES):
            self.V_prev = self.V_PREV_THRES # constant moved into constructor
        
        # compute virtual controls
        u1 = xdd_d + self.k_p * (x_d - state.x) + self.k_d * (xd_d - self.V_prev * np.cos(state.theta))
        u2 = ydd_d + self.k_p * (y_d - state.y) + self.k_d * (yd_d - self.V_prev * np.sin(state.theta))
        
        # np.linalg.solve --> J matrix --> then get actaul controls (a and omega)
        J = np.array([[np.cos(state.theta), -self.V_prev * np.sin(state.theta)], [np.sin(state.theta), self.V_prev * np.cos(state.theta)]])
        control_inputs = np.linalg.solve(J, np.array([u1, u2]))
        V_control = control_inputs[0]*dt + self.V_prev
        omega = control_inputs[1]
        
        # can also optionally calculate as such
        # V_control = self.V_prev + dt * (u1 * np.cos(state.theta) + u2 * np.sin(state.theta))
        # omega = (u2 * np.cos(state.theta) - u1 * np.sin(state.theta)) / V_control
        
        # No need to add clip / limit controls, since Base Navigator has built-in logic
        
        # Save current values to previous values for next function execution
        self.t_prev = t
        self.V_prev = V_control
        self.om_prev = omega
        
        # Create control message and return control values
        control_msg = TurtleBotControl()
        control_msg.v = V_control
        control_msg.omega = omega
        
        return control_msg
    
    
    def compute_trajectory_plan(self, state: TurtleBotState, goal: TurtleBotState,
                                occupancy: DetOccupancyGrid2D, resolution: float,
                                horizon: float) -> TrajectoryPlan:
        
        initial_state = (state.x, state.y)
        goal_state = (goal.x, goal.y)
        
        # (a) construct an AStar problem
        astar = AStar((0, 0), (horizon,horizon), initial_state, goal_state, occupancy, resolution=resolution)
        
        # (b) solve the problem
        if (not astar.solve()):
            return None
        
        # (c) access the solution path
        path = np.asarray(astar.path)
        
        # 2. See sim_astar.ipynb for examples on how to check if a solution exists.
        if (np.shape(path)[0] < 4): # check if 
            return None
        
        # 3. The compute_trajectory_tracking_control method uses some class properties to keep track of the ODE integration states. 
        #    What are those variables? How should we reset them when a new plan is generated?
        self.V_prev = 0.0
        self.om_prev = 0.0
        self.t_prev = 0.0
        ts = np.zeros(len(path))
        path_x = np.array([])
        path_y = np.array([])
        cumulative = 0
        
        # 4. See compute_smooth_plan function from sim_astar.ipynb.         
        # Separate path into x and y components
        for i in range(0, len(path)):
            path_x = np.append(path_x, path[i][0])
            path_y = np.append(path_y, path[i][1])
        
        # Calculate cumulative time for each waypoint
        delta_t = np.zeros(path.shape[0]) 
        # In order to compute time stamps for each way point, I will use the velocity formula: x = v * t --> t = x / v  
        # there is probably a vectorization method that can speed this compuation up
        for i in range(len(path) - 1):
            delta_t[i] = np.linalg.norm(path[i+1] - path[i], axis=-1) / self.V_desired + cumulative # axis = -1 needed to segregate tuples x and y coordinates norm calc 
            cumulative = ts[i+1]
    
        ts = np.cumsum(delta_t) # get timestamps from cumulative delta_t's
        
        # Fit cubic splines for x and y
        # must set k = 3 for cubic spline, spline_alpha is given in the previous function arguments
        spline_alpha = 0.05
        spline_x = splrep(ts, path[:, 0], k = 3, s = spline_alpha)
        spline_y = splrep(ts, path[:, 1], k = 3, s = spline_alpha)
        
        # 5. See the block in HW2 compute_smooth_plan on how to construct a TrajectoryPlan.
        
        return TrajectoryPlan(path, spline_x, spline_y, ts[-1])
    
if __name__ == "__main__":
    rclpy.init()            # initialize ROS client library
    nav_node = NavigationNode()    # create the node instance
    rclpy.spin(nav_node)        # call ROS2 default scheduler
    rclpy.shutdown()        # clean up after node exits
