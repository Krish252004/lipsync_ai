import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional

class FaceDetector:
    def __init__(self):
        """Initialize face detection model from MediaPipe."""
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5
        )

    def detect_faces(self, frame: np.ndarray) -> list:
        """Detect faces in a frame.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            List of detected faces with bounding boxes
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        if not results.detections:
            return []
            
        faces = []
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            faces.append((x, y, width, height))
            
        return faces

    def get_facial_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Get facial landmarks using MediaPipe FaceMesh.
        
        Args:
            frame: Input image in BGR format
            
        Returns:
            Numpy array of facial landmarks or None if no face detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])
            
        return np.array(landmarks)

    def release(self):
        """Release resources."""
        self.face_detection.close()
        self.face_mesh.close()