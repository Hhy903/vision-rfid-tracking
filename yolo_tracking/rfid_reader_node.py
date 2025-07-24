import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3
import mercury
import time
import math
import glob

class RFIDReaderNode(Node):
    def __init__(self):
        super().__init__('rfid_reader_node')
        self.publisher = self.create_publisher(Vector3, '/rfid_signal', 10)
        self.target_epc = "E2000017221101421890XXXX"  # ← 替换为你需要跟踪的 EPC
        self.reader = None
        self.setup_reader()

    def setup_reader(self):
        devices = glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")
        for device in devices:
            try:
                self.reader = mercury.Reader(f"tmr://{device}", baudrate=115200)
                self.reader.set_region("NA")
                self.reader.set_read_plan([1], "GEN2", read_power=2000)
                self.get_logger().info(f"Connected to RFID reader at {device}")
                self.reader.start_reading(self.tag_callback)
                return
            except Exception as e:
                self.get_logger().warn(f"Failed to connect {device}: {e}")

        self.get_logger().error("No RFID reader found!")

    def tag_callback(self, tag):
        epc = tag.epc.hex().upper()
        if epc != self.target_epc:
            return  # 不匹配就忽略

        rssi = tag.rssi
        angle_rad = 0.0  # 可扩展为相位转换
        distance = 1.0 / (abs(rssi) + 1e-3)  # 简易估距

        msg = Vector3()
        msg.x = angle_rad
        msg.y = distance
        msg.z = rssi
        self.publisher.publish(msg)

        self.get_logger().info(f"Matched EPC: {epc} | RSSI: {rssi:.1f} | Distance: {distance:.2f}")

def main(args=None):
    rclpy.init(args=args)
    node = RFIDReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.reader:
            node.reader.stop_reading()
        node.destroy_node()
        rclpy.shutdown()
        print("RFID reader stopped.")