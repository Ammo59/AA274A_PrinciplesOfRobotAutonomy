#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription(
        [
            # Declare use_sim_time launch argument with default value "true"
            DeclareLaunchArgument("use_sim_time", default_value="true"),

            # Include rviz.launch.py from package asl_tb3_sim
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
                    )
                ),
                launch_arguments={
                    "config": PathJoinSubstitution(
                        [
                            FindPackageShare("autonomy_repo"),
                            "rviz",
                            "default.rviz",
                        ]
                    ),
                    "use_sim_time": use_sim_time,
                }.items(),
            ),

            # Node for relaying RVIZ goal pose with updated output channel
            Node(
                executable="rviz_goal_relay.py",
                package="asl_tb3_lib",
                parameters=[
                    {"output_channel": "/cmd_nav"},  # Changed to /cmd_nav as required
                ],
            ),

            # State publisher node
            Node(
                executable="state_publisher.py",
                package="asl_tb3_lib",
            ),

            # Navigator node with use_sim_time parameter
            Node(
                executable="navigator.py",  # Changed to navigator.py
                package="autonomy_repo",  # Changed package to autonomy_repo
                parameters=[
                    {"use_sim_time": use_sim_time},  # Added use_sim_time parameter
                ],
            ),
        ]
    )
