import cv2
import numpy as np
from typing import Iterator, Tuple, Optional

class VideoProcessor:
    """Utility class for video processing operations."""
    
    def __init__(self, video_path: str, target_fps: int = 25):
        """Initialize video processor.
        
        Args:
            video_path: Path to video file
            target_fps: Target frames per second for processing
        """
        self.video_path = video_path
        self.target_fps = target_fps
        self.cap = None
        
    def __enter__(self):
        """Context manager entry."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.cap is not None:
            self.cap.release()
            
    def get_frames(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Generator that yields video frames with their timestamps.
        
        Yields:
            Tuple of (frame_index, frame) where frame is in BGR format
        """
        original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(round(original_fps / self.target_fps)))
        
        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            if frame_index % frame_interval == 0:
                yield frame_index, frame
                
            frame_index += 1
            
    def get_frame_count(self) -> int:
        """Get total number of frames in video."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_video_info(self) -> Tuple[int, int, int]:
        """Get video information (width, height, fps)."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return width, height, fps