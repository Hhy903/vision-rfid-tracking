import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
import csv
import os

class MovingAmbulance(Node):
    def __init__(self):
        super().__init__('moving_ambulance_node')
        self.cli = self.create_client(SetEntityState, 'set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('等待 /set_entity_state 服务中...')

        self.timer = self.create_timer(0.1, self.move_ambulance)
        self.time = 0.0

        # 正方形边长
        self.edge_length = 8.0
        self.edge_time = 40.0
        self.total_edges = 4
        self.start_x = 4.0
        self.start_y = 0.0

        # 保存轨迹
        self.trajectory = []
        self.output_path = '/tmp/ambulance_trajectory.csv'

        # 写入标题
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])

    def move_ambulance(self):
        state = EntityState()
        state.name = 'ambulance'

        t = self.time
        total_cycle_time = self.edge_time * self.total_edges
        t_mod = t % total_cycle_time
        edge_idx = int(t_mod // self.edge_time)
        t_edge = t_mod % self.edge_time
        speed = self.edge_length / self.edge_time  # 0.2 m/s

        if edge_idx == 0:
            x = self.start_x + speed * t_edge
            y = self.start_y
        elif edge_idx == 1:
            x = self.start_x + self.edge_length
            y = self.start_y + speed * t_edge
        elif edge_idx == 2:
            x = (self.start_x + self.edge_length) - speed * t_edge
            y = self.start_y + self.edge_length
        else:
            x = self.start_x
            y = (self.start_y + self.edge_length) - speed * t_edge

        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = 0.0
        state.pose.orientation.w = 1.0

        req = SetEntityState.Request()
        req.state = state
        self.cli.call_async(req)

        # 记录坐标
        self.trajectory.append((x, y))
        if len(self.trajectory) % 10 == 0:
            with open(self.output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.trajectory)
            self.trajectory.clear()

        self.time += 0.1

def main(args=None):
    rclpy.init(args=args)
    node = MovingAmbulance()
    rclpy.spin(node)
    rclpy.shutdown()
