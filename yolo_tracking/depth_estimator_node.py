#Use MiDaS model to estimate depth with single rgb image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import numpy as np
import cv2

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('depth_estimator_node')

        self.declare_parameter("debug", True)  
        self.debug_mode = self.get_parameter("debug").get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.pub_depth = self.create_publisher(Image, '/depth_map', 10)
        self.pub_mask = self.create_publisher(Image, '/safe_mask', 10)
        self.pub_colormap = self.create_publisher(Image, '/depth_colormap', 10)

        self.device = torch.device("cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device).eval()

        self.SAFE_DEPTH = 0.5
        self.BLOCK_SIZE = 16

    def simple_transform(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (160, 128))
        img = img / 255.0
        return torch.from_numpy(img.transpose(2,0,1)).float().unsqueeze(0).to(self.device)

    def get_safe_mask(self, depth_map):
        h, w = depth_map.shape
        grid_h = h // self.BLOCK_SIZE
        grid_w = w // self.BLOCK_SIZE
        safe_mask = np.zeros((grid_h, grid_w), dtype=np.uint8)

        for i in range(grid_h):
            for j in range(grid_w):
                y1 = i*self.BLOCK_SIZE
                y2 = (i+1)*self.BLOCK_SIZE
                x1 = j*self.BLOCK_SIZE
                x2 = (j+1)*self.BLOCK_SIZE
                block_depth = np.mean(depth_map[y1:y2, x1:x2])
                safe_mask[i,j] = 255 if block_depth > self.SAFE_DEPTH else 0

        return cv2.resize(cv2.dilate(safe_mask, None), (w, h))

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        input_tensor = self.simple_transform(frame)

        with torch.no_grad():
            depth = self.model(input_tensor)
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),
                size=(240, 320),
                mode="bicubic",
                align_corners=False
            ).squeeze().cpu().numpy()

        depth_map = cv2.normalize(depth, None, 0, 1, cv2.NORM_MINMAX)
        safe_mask = self.get_safe_mask(depth_map)

        # 发布深度图
        depth_img = (depth_map * 255).astype(np.uint8)
        depth_msg = self.bridge.cv2_to_imgmsg(depth_img, encoding='mono8')
        self.pub_depth.publish(depth_msg)

        # 发布安全掩码图
        mask_msg = self.bridge.cv2_to_imgmsg(safe_mask, encoding='mono8')
        self.pub_mask.publish(mask_msg)

        # 发布伪彩色深度图（CV2 彩色图）
        colormap = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
        color_msg = self.bridge.cv2_to_imgmsg(colormap, encoding='bgr8')
        self.pub_colormap.publish(color_msg)

        # ✅ OpenCV 弹窗可视化（仅调试模式开启）
        if self.debug_mode:
            # 可通行区域（绿色） + 不可通行区域（红色）
            overlay = frame.copy()
            overlay[safe_mask == 255] = (0, 255, 0)   # 绿色
            overlay[safe_mask == 0] = (0, 0, 255)     # 红色
            mixed = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            cv2.imshow("Depth Debug View", mixed)
            key = cv2.waitKey(1)
            if key == ord('q'):
                cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = DepthEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()