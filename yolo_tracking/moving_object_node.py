import rclpy
from rclpy.node import Node
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
import math

class MovingObject(Node):
    def __init__(self):
        super().__init__('moving_object_node')
        self.cli = self.create_client(SetModelState, '/gazebo/set_model_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('⏳ 等待 /gazebo/set_model_state 服务中...')

        self.timer = self.create_timer(0.1, self.timer_callback)
        self.t = 0.0

    def timer_callback(self):
        self.t += 0.1
        x = 1.5 + 1.0 * math.sin(self.t)  # 来回运动
        y = 0.0
        z = 0.1

        state = ModelState()
        state.model_name = 'moving_box'  # ⬅️ 你插入模型时要手动改名为 moving_box
        state.pose.position.x = x
        state.pose.position.y = y
        state.pose.position.z = z
        state.pose.orientation.w = 1.0

        req = SetModelState.Request()
        req.model_state = state

        self.cli.call_async(req)

def main(args=None):
    rclpy.init(args=args)
    node = MovingObject()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
