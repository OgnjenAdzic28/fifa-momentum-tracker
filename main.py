#!/usr/bin/env python3

import threading
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, Optional

from src.capture.screen_capture import ScreenCapture
from src.capture.controller_input import ControllerInputLogger
from src.detection.momentum_detector import MomentumDetector
from src.automation.auto_pause import AutoPauseSystem
from src.analytics.performance_tracker import PerformanceTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/fifa_momentum.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class FIFAMomentumTracker:
    def __init__(self):
        self.screen_capture = ScreenCapture()
        self.controller_logger = ControllerInputLogger()
        self.momentum_detector = MomentumDetector()
        self.auto_pause = AutoPauseSystem()
        self.performance_tracker = PerformanceTracker()
        
        self.is_running = False
        self.current_match_id = None
        
        # Load pre-trained model if available
        try:
            self.momentum_detector.load_model("models/momentum_model.pkl")
        except:
            logger.warning("No pre-trained model found. Using rule-based detection.")
    
    def start_tracking(self):
        self.is_running = True
        self.current_match_id = f"match_{int(time.time())}"
        
        logger.info("🚀 FIFA Momentum Tracker started!")
        logger.info("🎮 Make sure FIFA is running and in focus")
        
        # Start controller logging
        self.controller_logger.start_logging()
        
        # Start screen capture with callback
        capture_thread = threading.Thread(
            target=self.screen_capture.start_continuous_capture,
            args=[self.process_frame]
        )
        capture_thread.start()
        
        try:
            # Main monitoring loop
            while self.is_running:
                time.sleep(0.1)  # Small delay to prevent high CPU usage
                
        except KeyboardInterrupt:
            logger.info("Stopping FIFA Momentum Tracker...")
            self.stop_tracking()
    
    def process_frame(self, frame, timestamp):
        if not self.is_running:
            return
        
        try:
            # Extract visual features from the frame
            visual_features = self.momentum_detector.extract_visual_features(frame)
            
            # Get controller data
            controller_state = self.controller_logger.get_current_state()
            controller_data = controller_state.__dict__ if controller_state else None
            
            # Create momentum features
            momentum_features = self.momentum_detector.create_momentum_features(
                visual_features, controller_data
            )
            
            # Predict momentum
            momentum_prediction = self.momentum_detector.predict_momentum(momentum_features)
            
            # Check if we should trigger auto-pause
            player_stamina = self._estimate_player_stamina()  # Placeholder
            should_pause, reasons = self.auto_pause.should_trigger_pause(
                momentum_prediction, player_stamina
            )
            
            if should_pause:
                self._handle_momentum_intervention(momentum_prediction, reasons)
            
            # Log momentum data for analysis
            if hasattr(self, '_last_log_time'):
                if timestamp - self._last_log_time > 5.0:  # Log every 5 seconds
                    self._log_momentum_data(momentum_prediction)
                    self._last_log_time = timestamp
            else:
                self._last_log_time = timestamp
                
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
    
    def _handle_momentum_intervention(self, momentum_prediction, reasons):
        logger.warning(f"🔴 Momentum intervention triggered: {', '.join(reasons)}")
        
        # Execute auto-pause
        if self.auto_pause.execute_pause():
            # Generate recommendations
            player_stamina = self._estimate_player_stamina()
            substitutions = self.auto_pause.generate_substitution_recommendations(
                momentum_prediction, player_stamina
            )
            tactics = self.auto_pause.generate_tactical_recommendations(momentum_prediction)
            
            # Display recommendations to user
            self.auto_pause.display_recommendations(substitutions, tactics)
            
            # Wait for user input
            input("Press Enter when you've made your changes...")
            
            # Resume game
            self.auto_pause.resume_game()
            
            logger.info("✅ Intervention completed, game resumed")
    
    def _estimate_player_stamina(self) -> Dict[str, float]:
        # Placeholder stamina estimation
        # In a real implementation, this would analyze player movement patterns
        return {
            "Benzema": 0.7,
            "Modric": 0.4,
            "Vinicius": 0.8,
            "Casemiro": 0.3,
            "Carvajal": 0.6
        }
    
    def _log_momentum_data(self, momentum_prediction):
        logger.info(f"📊 Momentum: {momentum_prediction.momentum_level} "
                   f"(confidence: {momentum_prediction.confidence:.2f})")
    
    def stop_tracking(self):
        self.is_running = False
        self.screen_capture.stop_capture()
        self.controller_logger.stop_logging()
        
        # Save session data
        session_filename = f"data/raw/session_{int(time.time())}.json"
        self.controller_logger.save_session_data(session_filename)
        
        logger.info("📁 Session data saved")
        logger.info("👋 FIFA Momentum Tracker stopped")

def main():
    parser = argparse.ArgumentParser(description="FIFA Momentum Detection System")
    parser.add_argument("--train", action="store_true", help="Train the momentum detection model")
    parser.add_argument("--data", type=str, help="Path to training data CSV")
    parser.add_argument("--report", action="store_true", help="Generate performance report")
    parser.add_argument("--days", type=int, default=30, help="Days to include in report")
    
    args = parser.parse_args()
    
    if args.train:
        if not args.data:
            print("Error: --data path required for training")
            return
        
        detector = MomentumDetector()
        detector.train_model(args.data)
        detector.save_model("models/momentum_model.pkl")
        print("✅ Model training completed")
        return
    
    if args.report:
        tracker = PerformanceTracker()
        report = tracker.generate_performance_report(args.days)
        
        print("\n" + "="*60)
        print("📈 FIFA MOMENTUM TRACKER - PERFORMANCE REPORT")
        print("="*60)
        print(report['summary'])
        print(f"\n📊 Win Rate: {report['current_metrics']['win_rate']*100:.1f}%")
        print(f"⚽ Avg Goals Scored: {report['current_metrics']['avg_goals_scored']:.1f}")
        print(f"🛡️ Avg Goals Conceded: {report['current_metrics']['avg_goals_conceded']:.1f}")
        print("="*60)
        
        # Generate charts
        tracker.create_performance_charts()
        print("📈 Performance charts saved to data/charts/")
        return
    
    # Default: Start tracking
    tracker = FIFAMomentumTracker()
    tracker.start_tracking()

if __name__ == "__main__":
    main()