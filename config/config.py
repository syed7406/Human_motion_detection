import os

class Config:
    # Model Configuration
    MODEL_PATH = "yolov8n-pose.pt"  # or yolov8s.pt, yolov8m.pt for better accuracy
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4
    
    # Motion Detection Parameters
    MOTION_THRESHOLD = 30  # Minimum pixel movement to consider as motion
    MIN_CONTOUR_AREA = 500  # Minimum area to detect motion
    FRAME_BUFFER_SIZE = 10  # Frames to analyze for motion
    
    # Tracking Configuration
    MAX_TRACKING_AGE = 30  # Frames to keep tracking lost objects
    MIN_HITS = 3  # Minimum detections before confirming track
    
    # Visualization
    BOX_COLOR = (0, 255, 0)  # Green
    MOTION_COLOR = (0, 0, 255)  # Red
    TEXT_COLOR = (255, 255, 255)  # White
    TRAIL_COLOR = (255, 0, 255)  # Magenta
    THICKNESS = 2
    FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    
    # Video Configuration
    INPUT_SOURCE = 0  # 0 for webcam, or path to video file
    OUTPUT_PATH = "data/output/motion_detection.avi"
    SAVE_OUTPUT = True
    DISPLAY_FPS = True
    
    # Class filter (COCO dataset classes)
    PERSON_CLASS_ID = 0  # Only detect humans
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
    
    @classmethod
    def setup_directories(cls):
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)