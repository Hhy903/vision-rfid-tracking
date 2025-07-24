import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point, Vector3
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from collections import deque
import csv

class FusionController(Node):
    def __init__(self):
        super().__init__('fusion_controller_node')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub_target = self.create_subscription(Point, '/target_position', self.target_callback, 10)
        self.sub_rfid = self.create_subscription(Vector3, '/rfid_signal', self.rfid_callback, 10)
        self.sub_status = self.create_subscription(String, '/tracking_status', self.status_callback, 10)
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        self.target_pos = None
        self.rfid_data = None
        self.tracking_state = "Lost"
        self.error_buffer = deque(maxlen=5)

        self.robot_trajectory = []
        self.output_path = '/tmp/robot_trajectory.csv'
        with open(self.output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['x', 'y'])

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("融合控制节点启动完成")

    def target_callback(self, msg):
        self.target_pos = msg

    def rfid_callback(self, msg):
        self.rfid_data = msg

    def status_callback(self, msg):
        self.tracking_state = msg.data

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_trajectory.append((x, y))

        if len(self.robot_trajectory) % 10 == 0:
            with open(self.output_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(self.robot_trajectory)
            self.robot_trajectory.clear()

    def control_loop(self):
        cmd = Twist()

        if self.tracking_state == "Tracking" and self.target_pos is not None:
            error_x = self.target_pos.x - 320.0
            self.error_buffer.append(error_x)
            smooth_error = sum(self.error_buffer) / len(self.error_buffer)
            area = self.target_pos.z

            gain = 0.0005 if abs(smooth_error) < 80 else 0.005
            angular_z = -gain * smooth_error
            angular_z = max(min(angular_z, 0.2), -0.2)
            linear_x = 0.1 if area < 80000 and abs(smooth_error) < 80 else 0.0

            cmd.linear.x = linear_x
            cmd.angular.z = angular_z
            self.get_logger().info(f"[Tracking] 偏移={smooth_error:.1f}, 区域={area:.1f} → linear={linear_x:.2f}, angular={angular_z:.2f}")

        elif self.tracking_state == "Predicting":
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("[Predicting] 暂停控制，等待视觉确认...")

        elif self.tracking_state == "Lost" and self.rfid_data is not None:
            angle = self.rfid_data.x
            distance = self.rfid_data.y
            rssi = self.rfid_data.z

            angular_z = max(min(angle, 0.3), -0.3)
            linear_x = 0.05 if distance > 0.4 else 0.0

            cmd.linear.x = linear_x
            cmd.angular.z = angular_z
            self.get_logger().info(f"[Lost→RFID] 角度={angle:.2f}, 距离={distance:.2f} → linear={linear_x:.2f}, angular={angular_z:.2f}")

        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info("[感知中断] 无目标与信号 → 停止")

        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = FusionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
