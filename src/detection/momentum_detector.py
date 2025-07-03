import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import cv2
import pickle
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MomentumFeatures:
    cpu_aggression_level: float
    player_speed_variance: float
    teammate_positioning_score: float
    shot_accuracy_drop: float
    pass_completion_rate: float
    possession_loss_frequency: float
    referee_bias_score: float
    stamina_drain_rate: float

@dataclass
class MomentumPrediction:
    momentum_level: str  # 'positive', 'neutral', 'negative'
    confidence: float
    features_used: MomentumFeatures
    timestamp: float

class MomentumDetector:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history: List[MomentumFeatures] = []
        
    def extract_visual_features(self, frame: np.ndarray) -> Dict[str, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Player detection using template matching (simplified)
        player_regions = self._detect_players(gray)
        
        # Calculate movement patterns
        speed_variance = self._calculate_speed_variance(player_regions)
        
        # Analyze positioning patterns
        positioning_score = self._analyze_positioning(player_regions)
        
        # Detect UI elements for score/stats
        ui_features = self._extract_ui_features(frame)
        
        return {
            'speed_variance': speed_variance,
            'positioning_score': positioning_score,
            **ui_features
        }
    
    def _detect_players(self, gray_frame: np.ndarray) -> List[Tuple[int, int]]:
        # Simplified player detection using contours
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        player_positions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Filter by size
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    player_positions.append((cx, cy))
        
        return player_positions
    
    def _calculate_speed_variance(self, player_positions: List[Tuple[int, int]]) -> float:
        if len(self.feature_history) < 2:
            return 0.0
        
        # Calculate movement between frames
        movements = []
        for i, pos in enumerate(player_positions):
            if i < len(self.feature_history[-1].teammate_positioning_score):
                # Simplified movement calculation
                movements.append(np.random.normal(1.0, 0.3))  # Placeholder
        
        return np.var(movements) if movements else 0.0
    
    def _analyze_positioning(self, player_positions: List[Tuple[int, int]]) -> float:
        if len(player_positions) < 2:
            return 0.5
        
        # Calculate average distance between players
        distances = []
        for i, pos1 in enumerate(player_positions):
            for j, pos2 in enumerate(player_positions[i+1:], i+1):
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                distances.append(dist)
        
        return np.mean(distances) / 1000.0 if distances else 0.5  # Normalized
    
    def _extract_ui_features(self, frame: np.ndarray) -> Dict[str, float]:
        # Extract game UI elements (score, stamina bars, etc.)
        h, w = frame.shape[:2]
        
        # Sample regions for UI elements
        score_region = frame[0:int(h*0.1), int(w*0.4):int(w*0.6)]
        stamina_region = frame[int(h*0.8):h, 0:int(w*0.2)]
        
        return {
            'ui_brightness': np.mean(cv2.cvtColor(score_region, cv2.COLOR_BGR2GRAY)),
            'stamina_level': np.mean(cv2.cvtColor(stamina_region, cv2.COLOR_BGR2GRAY))
        }
    
    def create_momentum_features(self, visual_data: Dict[str, float], 
                               controller_data: Optional[Dict] = None) -> MomentumFeatures:
        # Combine visual and controller data into momentum features
        return MomentumFeatures(
            cpu_aggression_level=visual_data.get('speed_variance', 0.5) * 2.0,
            player_speed_variance=visual_data.get('speed_variance', 0.5),
            teammate_positioning_score=visual_data.get('positioning_score', 0.5),
            shot_accuracy_drop=np.random.uniform(0, 1),  # Placeholder
            pass_completion_rate=np.random.uniform(0.6, 1.0),  # Placeholder
            possession_loss_frequency=np.random.uniform(0, 0.5),  # Placeholder
            referee_bias_score=np.random.uniform(0, 1),  # Placeholder
            stamina_drain_rate=visual_data.get('stamina_level', 128) / 255.0
        )
    
    def predict_momentum(self, features: MomentumFeatures) -> MomentumPrediction:
        if not self.is_trained:
            # Use rule-based system when model isn't trained
            momentum_score = (
                features.cpu_aggression_level * 0.3 +
                features.player_speed_variance * 0.2 +
                (1 - features.pass_completion_rate) * 0.2 +
                features.possession_loss_frequency * 0.3
            )
            
            if momentum_score > 0.7:
                momentum_level = 'negative'
                confidence = min(momentum_score, 0.9)
            elif momentum_score < 0.3:
                momentum_level = 'positive'
                confidence = min(1 - momentum_score, 0.9)
            else:
                momentum_level = 'neutral'
                confidence = 0.6
        else:
            # Use trained model
            feature_array = np.array([[
                features.cpu_aggression_level,
                features.player_speed_variance,
                features.teammate_positioning_score,
                features.shot_accuracy_drop,
                features.pass_completion_rate,
                features.possession_loss_frequency,
                features.referee_bias_score,
                features.stamina_drain_rate
            ]])
            
            feature_array = self.scaler.transform(feature_array)
            prediction = self.model.predict(feature_array)[0]
            confidence = np.max(self.model.predict_proba(feature_array))
            
            momentum_level = prediction
        
        return MomentumPrediction(
            momentum_level=momentum_level,
            confidence=confidence,
            features_used=features,
            timestamp=pd.Timestamp.now().timestamp()
        )
    
    def train_model(self, training_data_path: str):
        # Load training data and train the model
        try:
            data = pd.read_csv(training_data_path)
            
            feature_columns = [
                'cpu_aggression_level', 'player_speed_variance', 'teammate_positioning_score',
                'shot_accuracy_drop', 'pass_completion_rate', 'possession_loss_frequency',
                'referee_bias_score', 'stamina_drain_rate'
            ]
            
            X = data[feature_columns]
            y = data['momentum_level']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Model trained with accuracy: {accuracy:.3f}")
            logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            self.is_trained = True
            
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
    
    def save_model(self, filepath: str):
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")