import cv2
import numpy as np
import mss
import time
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class ScreenCapture:
    def __init__(self, monitor_index: int = 1, target_fps: int = 30):
        self.monitor_index = monitor_index
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps
        self.sct = mss.mss()
        self.is_capturing = False
        
    def get_monitor_info(self) -> Dict[str, Any]:
        return self.sct.monitors[self.monitor_index]
    
    def capture_frame(self) -> Optional[np.ndarray]:
        try:
            monitor = self.sct.monitors[self.monitor_index]
            screenshot = self.sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            return frame
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None
    
    def start_continuous_capture(self, callback_func):
        self.is_capturing = True
        last_time = time.time()
        
        while self.is_capturing:
            current_time = time.time()
            if current_time - last_time >= self.frame_delay:
                frame = self.capture_frame()
                if frame is not None:
                    callback_func(frame, current_time)
                last_time = current_time
            
            time.sleep(0.001)
    
    def stop_capture(self):
        self.is_capturing = False
    
    def get_game_region(self, frame: np.ndarray) -> Tuple[int, int, int, int]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, x + w, y + h)
        
        return (0, 0, frame.shape[1], frame.shape[0])