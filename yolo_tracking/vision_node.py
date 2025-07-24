import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
from .module_vision import RobustTracker

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)

        self.publisher = self.create_publisher(Point, '/target_position', 10)
        self.status_pub = self.create_publisher(String, '/tracking_status', 10)

        self.model = YOLO("yolov8n.pt")
        self.tracker = None
        self.trace = []

        self.selected_target = None
        self.tracking = False

        # 添加滑动窗口滤波缓冲
        self.x_buffer = deque(maxlen=5)
        self.z_buffer = deque(maxlen=5)

        cv2.namedWindow("YOLOv8 Tracking")
        cv2.setMouseCallback("YOLOv8 Tracking", self.mouse_callback)

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selected_target = (x, y)
            self.tracking = False
            print(f"[鼠标点击] Selected target at: {self.selected_target}")

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480)) 
        results = self.model(frame, conf=0.25)
        boxes = results[0].boxes.xywh.cpu().numpy()

        for box in boxes:
            x, y, w, h = box
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

        # 点击选择逻辑
        if self.selected_target is not None and not self.tracking:
            print(f"[匹配中] 尝试匹配点击点 {self.selected_target} 到目标框")
            print(f"当前检测框数：{len(boxes)}")
            min_dist = float('inf')
            target_box = None
            for box in boxes:
                x, y, w, h = box
                center = (int(x), int(y))
                dist = np.sqrt((center[0]-self.selected_target[0])**2 + (center[1]-self.selected_target[1])**2)
                print(f"检测框中心：{center}，距离点击点：{dist:.2f}")
                if dist < min_dist:
                    min_dist = dist
                    target_box = box
            if target_box is not None and min_dist < 200:
                print(f"[选中目标] 框：{target_box}，距离：{min_dist:.2f}")
                self.tracker = RobustTracker(target_box, frame)
                self.tracking = True
                self.selected_target = None
                self.trace = []
                self.x_buffer.clear()
                self.z_buffer.clear()
            else:
                print(f"[未命中] 最小距离为 {min_dist:.2f}，未进入跟踪状态")
                self.selected_target = None

        # 跟踪阶段
        if self.tracking and self.tracker is not None:
            pred_box = self.tracker.update(frame, boxes)
            x, y, w, h = pred_box
            center_x, center_y = int(x), int(y)
            area = w * h

            # 滑动窗口平滑
            self.x_buffer.append(center_x)
            self.z_buffer.append(area)

            smooth_x = sum(self.x_buffer) / len(self.x_buffer)
            smooth_z = sum(self.z_buffer) / len(self.z_buffer)

            # 绘图
            cv2.rectangle(frame,
                          (int(x - w/2), int(y - h/2)),
                          (int(x + w/2), int(y + h/2)),
                          (0, 0, 255), 2)

            # 发布目标位置
            point_msg = Point()
            point_msg.x = float(smooth_x)
            point_msg.y = float(center_y)
            point_msg.z = float(smooth_z)
            self.publisher.publish(point_msg)

            # 状态
            status_msg = String()
            if self.tracker.lost_count > 20:
                status_msg.data = "Lost"
            elif self.tracker.lost_count > 0:
                status_msg.data = "Predicting"
            else:
                status_msg.data = "Tracking"
            self.status_pub.publish(status_msg)

            # 轨迹
            self.trace.append((center_x, center_y))
            if len(self.trace) > 30:
                self.trace.pop(0)
            if len(self.trace) >= 2:
                pts = np.array(self.trace, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (255,0,0), 2)

            cv2.putText(frame, f"Status: {status_msg.data}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            cv2.putText(frame, "Click to select target", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if self.selected_target is not None:
            cv2.circle(frame, self.selected_target, 5, (255, 0, 255), -1)

        cv2.imshow("YOLOv8 Tracking", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
