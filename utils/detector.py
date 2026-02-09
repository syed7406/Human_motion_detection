# utils/detector.py
import cv2
import numpy as np
from ultralytics import YOLO

class HumanDetector:
    def __init__(self, model_path, confidence_threshold=0.5):
        self.model = YOLO(model_path)  # Now it's yolov8n-pose.pt
        self.confidence_threshold = confidence_threshold

    def detect_humans(self, frame):
        """
        Returns: List of humans with:
          - bbox 
          - keypoints (17 points)
          - detected movements (hand_raise, walking_left, sleeping, etc.)
        """
        results = self.model(frame, classes=[0], conf=self.confidence_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints  # <--- THIS IS NEW (POSE KEYPOINTS)

            for box_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Get 17 keypoints for this person (shape: [17, 3] -> x,y,conf)
                kp = keypoints.xy[box_idx].cpu().numpy()  # [17, 2] (x,y)
                kp_conf = keypoints.conf[box_idx].cpu().numpy()  # [17]

                # --- DETECT SPECIFIC MOVEMENTS ---
                movements = self._analyze_pose(kp, kp_conf, frame.shape[1])  # frame width for left/right

                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'movements': movements,  # ['hand_raise', 'walking_left', ...]
                    'keypoints': kp,         # For drawing skeleton
                    'keypoints_conf': kp_conf,
                })

        return detections

    ##################################################################
    # NEW: Analyze pose to detect hand raise, sleeping, left/right, etc.
    ##################################################################
    def _analyze_pose(self, keypoints, confidences, frame_width):
        movements = []

        # Keypoint indices (YOLOv8-pose)
        NOSE = 0
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        LEFT_HIP = 11
        RIGHT_HIP = 12

        # --- 1. HAND RAISE DETECTION ---
        if confidences[LEFT_WRIST] > 0.4 and confidences[LEFT_SHOULDER] > 0.4:
            wrist_y = keypoints[LEFT_WRIST][1]
            shoulder_y = keypoints[LEFT_SHOULDER][1]
            if wrist_y < shoulder_y - 20:  # Wrist above shoulder
                movements.append('hand_raise_left')

        if confidences[RIGHT_WRIST] > 0.4 and confidences[RIGHT_SHOULDER] > 0.4:
            wrist_y = keypoints[RIGHT_WRIST][1]
            shoulder_y = keypoints[RIGHT_SHOULDER][1]
            if wrist_y < shoulder_y - 20:
                movements.append('hand_raise_right')

        # --- 2. SLEEPING / LYING DOWN ---
        if confidences[LEFT_SHOULDER] > 0.4 and confidences[LEFT_HIP] > 0.4:
            shoulder = keypoints[LEFT_SHOULDER]
            hip = keypoints[LEFT_HIP]
            # If body is horizontal (slope near 0)
            slope = abs(shoulder[1] - hip[1]) 
            if slope < 10:  # Almost same Y-level → lying/sleeping
                movements.append('sleeping')

        # --- 3. LEFT/RIGHT MOVEMENT ---
        # Use shoulder midpoint as reference
        if confidences[LEFT_SHOULDER] > 0.4 and confidences[RIGHT_SHOULDER] > 0.4:
            mid_shoulder = (keypoints[LEFT_SHOULDER] + keypoints[RIGHT_SHOULDER]) / 2
            x_center = mid_shoulder[0]

            '''if x_center < frame_width * 0.45:
                movements.append('walking_left')
            elif x_center > frame_width * 0.55:
                movements.append('walking_right')'''

        # --- 4. SITTING vs STANDING ---
        if confidences[LEFT_HIP] > 0.4 and confidences[LEFT_SHOULDER] > 0.4:
            hip_y = keypoints[LEFT_HIP][1]
            shoulder_y = keypoints[LEFT_SHOULDER][1]
            body_height = abs(shoulder_y - hip_y)

            if body_height < 60:  # Relative threshold (adjust for your camera)
                movements.append('sitting')
            else:
                movements.append('standing')

        return movements