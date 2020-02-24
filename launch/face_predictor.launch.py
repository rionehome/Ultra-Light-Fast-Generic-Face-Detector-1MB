from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import LogInfo


def generate_launch_description():
    return LaunchDescription([
        #############################################################
        LogInfo(
            msg="launch realsense_ros2_camera"
        ),
        Node(
            package="face_predictor",
            node_executable="face_predictor",
            output="screen",
        ),
    ])
