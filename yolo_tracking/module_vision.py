
#!/usr/bin/env python3


from collections import deque
import cv2
import numpy as np
from ultralytics import YOLO

def draw_legend_once(frame, color_info, start_x=10, start_y=90, font_scale=0.6, font_thickness=2):
    """
    绘制图例，用于显示各种状态及对应颜色
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    for i, (state, color) in enumerate(color_info.items()):
        cv2.rectangle(frame, (start_x, start_y + i * 30), (start_x + 20, start_y + i * 30 + 20), color, -1)
        text = f"{state}: {color}"
        cv2.putText(frame, text, (start_x + 30, start_y + i * 30 + 15),
                    font, font_scale, (255, 255, 255), font_thickness)

# -------------------------------------------
# 1. 改进型卡尔曼滤波器
# -------------------------------------------
class EnhancedKalmanFilter:
    def __init__(self, initial_box):
        # 状态维度：8 (x, y, w, h, dx, dy, ddx, ddy)
        self.kf = cv2.KalmanFilter(8, 4)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0.5, 0],
            [0, 1, 0, 0, 0, 1, 0, 0.5],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.5

        self.kf.statePost = np.array([
            initial_box[0], initial_box[1],
            initial_box[2], initial_box[3],
            0, 0,
            0, 0
        ], dtype=np.float32).reshape(8, 1)

    def predict(self):
        return self.kf.predict()

    def correct(self, measurement):
        self.kf.correct(measurement.reshape(4, 1))

# -------------------------------------------
# 2. 数据关联辅助函数
# -------------------------------------------
def calculate_iou(box1, box2):
    """计算两个边界框的IOU，box格式为 [x_center, y_center, w, h]"""
    box1_coords = [box1[0]-box1[2]/2, box1[1]-box1[3]/2,
                   box1[0]+box1[2]/2, box1[1]+box1[3]/2]
    box2_coords = [box2[0]-box2[2]/2, box2[1]-box2[3]/2,
                   box2[0]+box2[2]/2, box2[1]+box2[3]/2]
    xA = max(box1_coords[0], box2_coords[0])
    yA = max(box1_coords[1], box2_coords[1])
    xB = min(box1_coords[2], box2_coords[2])
    yB = min(box1_coords[3], box2_coords[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1_coords[2] - box1_coords[0]) * (box1_coords[3] - box1_coords[1])
    box2_area = (box2_coords[2] - box2_coords[0]) * (box2_coords[3] - box2_coords[1])
    return inter_area / (box1_area + box2_area - inter_area + 1e-6)

def motion_consistency(pred_velocity, pred_center, meas_center):
    """计算运动方向一致性评分，返回0到1之间的值"""
    meas_vector = np.array(meas_center) - np.array(pred_center)
    if np.linalg.norm(meas_vector) < 1e-6:
        return 0.0
    pred_vector = pred_velocity[:2].flatten()
    cosine_sim = np.dot(pred_vector, meas_vector) / (np.linalg.norm(pred_vector) * np.linalg.norm(meas_vector) + 1e-6)
    return max(0.0, cosine_sim)

# -------------------------------------------
# 3. 特征匹配模块
# -------------------------------------------
class FeatureBank:
    def __init__(self, max_features=5):
        self.features = deque(maxlen=max_features)
    
    def add_feature(self, img_patch):
        """根据颜色直方图提取目标外观特征"""
        hist = cv2.calcHist([img_patch], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        self.features.append(hist)
    
    def match(self, img_patch):
        """计算当前图像块与特征库中的最大相关性得分"""
        query_hist = cv2.calcHist([img_patch], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
        query_hist = cv2.normalize(query_hist, query_hist).flatten()
        best_score = 0
        for hist in self.features:
            score = cv2.compareHist(query_hist, hist, cv2.HISTCMP_CORREL)
            best_score = max(best_score, score)
        return best_score if best_score > 0.85 else None

# -------------------------------------------
# 4. RobustTracker 类：整合卡尔曼滤波和特征匹配
# -------------------------------------------
class RobustTracker:
    def __init__(self, initial_box, initial_frame):
        self.kf = EnhancedKalmanFilter(initial_box)
        self.feature_bank = FeatureBank()
        self._update_feature(initial_frame, initial_box)
        self.lost_count = 0
        self.occlusion_threshold = 0.3   # IOU阈值
    
    def _update_feature(self, frame, box):
        x, y, w, h = box
        x1, y1 = int(x - w/2), int(y - h/2)
        x2, y2 = int(x + w/2), int(y + h/2)
        # 检查是否越界
        if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
            return
        patch = frame[y1:y2, x1:x2]
        if patch.size > 0:
            self.feature_bank.add_feature(patch)
    
    def update(self, frame, detections):
        pred_box = self.kf.predict()[:4].flatten()
        best_match = None
        best_score = -1

        # --- 阶段1：标准IOU匹配 ---
        for box in detections:
            iou = calculate_iou(pred_box, box)
            motion_score = motion_consistency(self.kf.kf.statePost[4:], pred_box[:2], box[:2])
            total_score = iou * 0.6 + motion_score * 0.4
            if total_score > best_score:
                best_score = total_score
                best_match = box

        if best_score >= self.occlusion_threshold:
            # 匹配成功
            self.kf.correct(best_match)
            self._update_feature(frame, best_match)
            self.lost_count = 0
            return self.kf.kf.statePost[:4].flatten()

        # --- 阶段2：尝试通过特征重识别恢复 ---
        for box in detections:
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            similarity = self.feature_bank.match(patch)
            if similarity is not None and similarity > 0.7:
                print("通过特征匹配重新捕获目标！")
                self.kf.correct(box)
                self._update_feature(frame, box)
                self.lost_count = 0
                return self.kf.kf.statePost[:4].flatten()

        # --- 阶段3：完全失联，执行预测，计数加1 ---
        self.lost_count += 1
        return pred_box


# -------------------------------------------
# 5. 鼠标点击回调辅助函数
# -------------------------------------------
# 这个函数用于在OpenCV窗口中捕捉鼠标点击事件，
# 并选择与点击点距离最近的box作为初始跟踪目标。
def select_target(event, x, y, flags, param):
    """
    全局变量 selected_target, tracking 在ROS2节点中可转化为类成员变量或服务接口，
    此处仅用于测试原始代码。
    """
    global selected_target, tracking
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_target = (x, y)
        tracking = False
        print(f"Selected target at: {selected_target}")

# -------------------------------------------
# 全局变量（调试时用于OpenCV窗口交互）
# -------------------------------------------
selected_target = None
tracking = False

# -------------------------------------------
# 测试主函数（在ROS2节点中会用图像回调代替此循环）
# -------------------------------------------
if __name__ == "__main__":
    # 加载YOLO模型
    model = YOLO("yolov8n.pt")
    
    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to open the camera!")
        exit()
    
    cv2.namedWindow("YOLOv8 Tracking")
    cv2.setMouseCallback("YOLOv8 Tracking", select_target)

    tracker = None
    trace = []
    
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera!")
            break

        results = model(frame)
        boxes = results[0].boxes.xywh.cpu().numpy()  # 格式：[x, y, w, h]

        # 绘制所有检测框（绿色框）
        for box in boxes:
            x, y, w, h = box
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        
        # 如果有鼠标点击且当前未跟踪，则根据点击点选择目标
        if selected_target is not None and not tracking:
            min_dist = float('inf')
            target_box = None
            for box in boxes:
                x, y, w, h = box
                center = (int(x), int(y))
                dist = np.sqrt((center[0]-selected_target[0])**2 + (center[1]-selected_target[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    target_box = box
            if target_box is not None and min_dist < 100:  # 距离阈值
                print(f"Target found! Box: {target_box}, Distance: {min_dist}")
                tracker = RobustTracker(target_box, frame)
                tracking = True
                trace = []
                # 重置选择
                selected_target = None
            else:
                print(f"No target found near the selected point. Closest distance: {min_dist}")
                selected_target = None
        
        # 如果进入跟踪状态，则更新tracker
        if tracking and tracker is not None:
            pred_box = tracker.update(frame, boxes)
            x, y, w, h = pred_box
            cv2.rectangle(frame,
                          (int(x - w/2), int(y - h/2)),
                          (int(x + w/2), int(y + h/2)),
                          (0, 0, 255), 2)
            trace.append((int(x), int(y)))
            if len(trace) > 30:
                trace.pop(0)
            if len(trace) >= 2:
                pts = np.array(trace, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (255,0,0), 2)
            status = "Tracking" if tracker.lost_count == 0 else "Predicting"
            cv2.putText(frame, f"Status: {status}", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        else:
            cv2.putText(frame, "Click to select target", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        
        cv2.imshow("YOLOv8 Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
