import cv2
import numpy as np
from typing import Tuple, List

class LipExtractor:
    """Extracts lip region from facial landmarks."""
    
    # Lip landmark indices in MediaPipe FaceMesh
    LOWER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    UPPER_LIP = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415]
    
    def __init__(self, target_size: Tuple[int, int] = (96, 96)):
        """Initialize lip extractor.
        
        Args:
            target_size: Target size (width, height) for extracted lip region
        """
        self.target_size = target_size
        
    def get_lip_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Get lip landmarks from full facial landmarks.
        
        Args:
            landmarks: Full facial landmarks from MediaPipe
            
        Returns:
            Lip landmarks only
        """
        lip_indices = self.LOWER_LIP + self.UPPER_LIP
        return landmarks[sorted(lip_indices)]
    
    def extract_lip_region(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """Extract and align lip region from frame.
        
        Args:
            frame: Input image in BGR format
            landmarks: Lip landmarks
            
        Returns:
            Extracted and aligned lip region
        """
        # Get bounding box of lip region
        h, w = frame.shape[:2]
        lip_landmarks = self.get_lip_landmarks(landmarks)
        
        # Convert normalized landmarks to pixel coordinates
        pixel_landmarks = np.zeros_like(lip_landmarks)
        pixel_landmarks[:, 0] = lip_landmarks[:, 0] * w
        pixel_landmarks[:, 1] = lip_landmarks[:, 1] * h
        
        # Calculate bounding box with some padding
        min_x, min_y = np.min(pixel_landmarks[:, :2], axis=0)
        max_x, max_y = np.max(pixel_landmarks[:, :2], axis=0)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Add padding
        padding = max(width, height) * 0.3
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)
        max_x = min(w, max_x + padding)
        max_y = min(h, max_y + padding)
        
        # Extract and resize lip region
        lip_region = frame[int(min_y):int(max_y), int(min_x):int(max_x)]
        lip_region = cv2.resize(lip_region, self.target_size)
        
        return lip_region