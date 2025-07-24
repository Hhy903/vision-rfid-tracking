import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Vector3
from gazebo_msgs.msg import ModelStates
import math

class RFIDSimulator(Node):
    def __init__(self):
        super().__init__('rfid_simulator')

        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.sub_models = self.create_subscription(ModelStates, '/model_states', self.model_states_callback, 10)
        self.pub_signal = self.create_publisher(Vector3, '/rfid_signal', 10)

        self.robot_pos = None
        self.target_pos = None

        # 目标模型的名字（确保与Gazebo中的名称一致）
        self.target_model_name = "ambulance"

        self.get_logger().info(f"[RFID模拟器] 等待目标模型 `{self.target_model_name}` 的位置...")

    def odom_callback(self, msg):
        self.robot_pos = msg.pose.pose.position

    def model_states_callback(self, msg):
        self.get_logger().info(f"[DEBUG] model_states_callback 被触发，模型列表: {msg.name}")
        if self.target_model_name in msg.name:
            idx = msg.name.index(self.target_model_name)
            self.target_pos = msg.pose[idx].position
            self.simulate_rssi()

    def simulate_rssi(self):
        if self.robot_pos is None or self.target_pos is None:
            return

        dx = self.target_pos.x - self.robot_pos.x
        dy = self.target_pos.y - self.robot_pos.y
        distance = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx)
        rssi = 1.0 / (distance**2 + 0.01)  # 防止除0

        signal_msg = Vector3()
        signal_msg.x = angle        # 相对角度
        signal_msg.y = distance     # 直线距离
        signal_msg.z = rssi         # RSSI 强度
        self.pub_signal.publish(signal_msg)

        self.get_logger().info(
            f"[RFID] 模拟信号 → 角度: {angle:.2f} rad, 距离: {distance:.2f}, RSSI: {rssi:.3f}"
        )

def main(args=None):
    rclpy.init(args=args)
    node = RFIDSimulator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
