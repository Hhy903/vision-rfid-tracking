from ament_index_python import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # 设置模型路径
    TURTLEBOT3_MODEL = 'waffle'
    TURTLEBOT3_MODEL_PATH = os.path.join(
        get_package_share_directory('turtlebot3_gazebo'),
        'launch',
        'turtlebot3_world.launch.py'
    )

    # 主控机器人命名空间
    robot1_namespace = 'robot1'
    # 目标机器人命名空间
    robot2_namespace = 'robot2'

    return LaunchDescription([
        # 启动 Gazebo 世界
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(TURTLEBOT3_MODEL_PATH),
            launch_arguments={'namespace': robot1_namespace}.items()
        ),
        # 启动第二台机器人
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(TURTLEBOT3_MODEL_PATH),
            launch_arguments={'namespace': robot2_namespace}.items()
        ),
        # 启动目标机器人的移动节点
        Node(
            package='yolo_tracking',
            executable='moving_target_node',
            name='moving_target_node',
            namespace=robot2_namespace,
            output='screen'
        ),
        # 启动主控机器人的控制节点
        Node(
            package='yolo_tracking',
            executable='tracking_controller_node',
            name='tracking_controller_node',
            namespace=robot1_namespace,
            output='screen'
        ),
    ])
