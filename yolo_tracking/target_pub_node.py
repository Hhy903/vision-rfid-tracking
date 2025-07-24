import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point


class TargetPublisher(Node):
    def __init__(self):
        super().__init__('target_publisher')
        self.pub = self.create_publisher(Point, '/target_position', 10)
        self.timer = self.create_timer(1.0, self.publish_target)

    def publish_target(self):
        target = Point()
        target.x = 2.0  # 固定目标坐标
        target.y = 1.0
        target.z = 0.0
        self.pub.publish(target)
        self.get_logger().info(f"发布目标点: ({target.x}, {target.y})")

def main(args=None):
    rclpy.init(args=args)
    node = TargetPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
