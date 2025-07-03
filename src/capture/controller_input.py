import pygame
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import threading
import logging

logger = logging.getLogger(__name__)

@dataclass
class ControllerState:
    timestamp: float
    buttons: Dict[str, bool]
    axes: Dict[str, float]
    triggers: Dict[str, float]

class ControllerInputLogger:
    def __init__(self, controller_index: int = 0):
        pygame.init()
        pygame.joystick.init()
        
        self.controller_index = controller_index
        self.joystick = None
        self.is_logging = False
        self.input_history: List[ControllerState] = []
        self.max_history = 1000
        
        self._initialize_controller()
    
    def _initialize_controller(self):
        if pygame.joystick.get_count() > self.controller_index:
            self.joystick = pygame.joystick.Joystick(self.controller_index)
            self.joystick.init()
            logger.info(f"Controller initialized: {self.joystick.get_name()}")
        else:
            logger.warning("No controller found")
    
    def get_current_state(self) -> Optional[ControllerState]:
        if not self.joystick:
            return None
        
        pygame.event.pump()
        
        buttons = {}
        for i in range(self.joystick.get_numbuttons()):
            buttons[f"button_{i}"] = bool(self.joystick.get_button(i))
        
        axes = {}
        for i in range(self.joystick.get_numaxes()):
            axes[f"axis_{i}"] = self.joystick.get_axis(i)
        
        triggers = {
            "left_trigger": axes.get("axis_2", 0.0),
            "right_trigger": axes.get("axis_5", 0.0)
        }
        
        return ControllerState(
            timestamp=time.time(),
            buttons=buttons,
            axes=axes,
            triggers=triggers
        )
    
    def start_logging(self):
        self.is_logging = True
        self.logging_thread = threading.Thread(target=self._logging_loop)
        self.logging_thread.start()
    
    def stop_logging(self):
        self.is_logging = False
        if hasattr(self, 'logging_thread'):
            self.logging_thread.join()
    
    def _logging_loop(self):
        while self.is_logging:
            state = self.get_current_state()
            if state:
                self.input_history.append(state)
                if len(self.input_history) > self.max_history:
                    self.input_history.pop(0)
            time.sleep(0.016)  # ~60 FPS
    
    def get_input_patterns(self, window_seconds: float = 5.0) -> List[ControllerState]:
        cutoff_time = time.time() - window_seconds
        return [state for state in self.input_history if state.timestamp >= cutoff_time]
    
    def detect_skill_moves(self) -> List[str]:
        recent_inputs = self.get_input_patterns(2.0)
        detected_moves = []
        
        # Simple skill move detection patterns
        for i, state in enumerate(recent_inputs[:-3]):
            if i < len(recent_inputs) - 3:
                next_states = recent_inputs[i+1:i+4]
                
                # Example: Ball roll (right stick flick)
                if (abs(state.axes.get("axis_3", 0)) > 0.8 and 
                    abs(next_states[0].axes.get("axis_3", 0)) < 0.2):
                    detected_moves.append("ball_roll")
        
        return detected_moves
    
    def save_session_data(self, filename: str):
        data = [asdict(state) for state in self.input_history]
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)