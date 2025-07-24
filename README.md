# Vision-RFID Target Tracking Robot

A ROS2-based mobile robot system for robust target tracking using YOLOv8 vision and RFID sensing.  
Designed for dynamic, cluttered environments, integrating visual detection, Kalman filtering, and RFID signal fusion.

## Features
- YOLOv8-based object detection and tracking
- RFID-guided re-identification for unique target following
- ROS2 nodes for perception, control, and sensor fusion

## Usage
### 1.Simulation
```bash
# Launch Gazebo
ros2 launch gazebo_ros gazebo.launch.py world:=trackingworld 
# Launch vision tracking node
ros2 run yolo_tracking vision_node 
# Launch RFID simulation node
ros2 run yolo_tracking rfid_sim_node.py
# Launch target node(an moving ambulance in gazebo)
ros2 run yolo_tracking moving_ambulance_node 
# Start fusion control node
ros2 run yolo_tracking fusion_tracking_node
```
### 2.Real world
Same as the simulation but you don't need to use **rfid_sim**node. Instead, launch **rfid_reader_node**. This node will use **python-mercuayapi** to read from the anthenna

Good luck!
