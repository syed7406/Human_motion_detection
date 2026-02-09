import numpy as np
from collections import deque
from scipy.spatial import distance
import uuid

class MotionTracker:
    """Advanced motion tracking for detected humans"""
    
    def __init__(self, max_disappeared=30, max_distance=50):
        self.next_object_id = 0
        self.objects = {}  # ID -> centroid
        self.disappeared = {}  # ID -> frames disappeared
        self.bboxes = {}  # ID -> bounding box
        self.motion_trails = {}  # ID -> deque of positions
        self.velocities = {}  # ID -> velocity vector
        self.motion_status = {}  # ID -> motion state
        self.detections = {}  # ID -> last detection dict
        
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trail_length = 30
        
    def register(self, centroid, bbox):
        """Register new tracked object"""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.disappeared[object_id] = 0
        self.motion_trails[object_id] = deque(maxlen=self.trail_length)
        self.motion_trails[object_id].append(centroid)
        self.velocities[object_id] = (0, 0)
        self.motion_status[object_id] = "Stationary"
        self.detections[object_id] = None
        self.next_object_id += 1
        
        return object_id
    
    def deregister(self, object_id):
        """Remove tracked object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        del self.bboxes[object_id]
        del self.motion_trails[object_id]
        del self.velocities[object_id]
        del self.motion_status[object_id]
        if object_id in self.detections:
            del self.detections[object_id]
    
    def update(self, detections):
        """
        Update tracker with new detections
        detections: list of (bbox, centroid, detection_dict) tuples
        """
        # No detections
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            return self.objects
        
        input_centroids = np.array([d[1] for d in detections])
        input_bboxes = [d[0] for d in detections]
        input_dets = [d[2] if len(d) > 2 else None for d in detections]
        
        # No existing objects - register all
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                obj_id = self.register(input_centroids[i], input_bboxes[i])
                # store detection dict if provided
                if input_dets[i] is not None:
                    # ensure stored centroids/bboxes are numpy arrays for consistency
                    self.detections[obj_id] = input_dets[i]
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance between existing and new centroids
            D = distance.cdist(np.array(object_centroids), input_centroids)

            # Match existing objects to new detections
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]

                # Calculate velocity
                old_centroid = self.objects[object_id]
                new_centroid = input_centroids[col]
                velocity = (
                    new_centroid[0] - old_centroid[0],
                    new_centroid[1] - old_centroid[1]
                )

                # Update object
                self.objects[object_id] = new_centroid
                self.bboxes[object_id] = input_bboxes[col]
                self.disappeared[object_id] = 0
                self.motion_trails[object_id].append(new_centroid)
                self.velocities[object_id] = velocity
                # store detection dict if provided
                if input_dets[col] is not None:
                    self.detections[object_id] = input_dets[col]

                # Determine movements/status
                movements = []
                if self.detections[object_id] is not None:
                    movements = self.detections[object_id].get('movements', [])

                if 'sleeping' in movements:
                    self.motion_status[object_id] = "Sleeping"
                elif any('hand_raise' in m for m in movements):
                    self.motion_status[object_id] = "Hand Raise"
                elif 'walking_left' in movements:
                    self.motion_status[object_id] = "Left"
                elif 'walking_right' in movements:
                    self.motion_status[object_id] = "Right"
                elif 'sitting' in movements:
                    self.motion_status[object_id] = "Sitting"
                else:
                    speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                    if speed < 2:
                        self.motion_status[object_id] = "Stationary"
                    elif speed < 10:
                        self.motion_status[object_id] = "Walking"
                    else:
                        self.motion_status[object_id] = "Running"

                used_rows.add(row)
                used_cols.add(col)
            
            # Mark disappeared objects
            unused_rows = set(range(D.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new objects
            unused_cols = set(range(D.shape[1])) - used_cols
            for col in unused_cols:
                obj_id = self.register(input_centroids[col], input_bboxes[col])
                if input_dets[col] is not None:
                    self.detections[obj_id] = input_dets[col]
        
        return self.objects
    
    def get_motion_info(self, object_id):
        """Get motion information for an object"""
        if object_id not in self.objects:
            return None
        
        return {
            'id': object_id,
            'position': self.objects[object_id],
            'bbox': self.bboxes[object_id],
            'velocity': self.velocities[object_id],
            'status': self.motion_status[object_id],
            'trail': list(self.motion_trails[object_id]),
            'detection': self.detections.get(object_id, None)
        }