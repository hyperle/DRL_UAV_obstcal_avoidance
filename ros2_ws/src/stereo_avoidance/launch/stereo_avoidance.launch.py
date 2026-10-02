from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="stereo_avoidance",
                executable="stereo_avoidance_node",
                name="stereo_avoidance_node",
                output="screen",
                parameters=[{"model_path": "depth_policy.onnx"}],
            )
        ]
    )
