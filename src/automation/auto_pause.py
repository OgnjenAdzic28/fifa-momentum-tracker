import pyautogui
import time
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SubstitutionReason(Enum):
    STAMINA_LOW = "stamina_low"
    MOMENTUM_NEGATIVE = "momentum_negative"
    TACTICAL_CHANGE = "tactical_change"
    INJURY_PREVENTION = "injury_prevention"

@dataclass
class PlayerSubstitution:
    player_out: str
    player_in: str
    position: str
    reason: SubstitutionReason
    priority: int  # 1-10, higher is more urgent

@dataclass
class TacticalAdjustment:
    formation_change: Optional[str]
    pressure_level: Optional[str]  # "low", "medium", "high"
    tempo: Optional[str]  # "slow", "balanced", "fast"
    width: Optional[str]  # "narrow", "balanced", "wide"

class AutoPauseSystem:
    def __init__(self):
        self.is_enabled = True
        self.pause_cooldown = 30  # seconds
        self.last_pause_time = 0
        self.pause_triggers = {
            'negative_momentum': True,
            'low_stamina': True,
            'tactical_adjustment': True
        }
        
        # FIFA-specific key bindings
        self.keybinds = {
            'pause': 'space',
            'substitutions': 'tab',
            'tactics': 't',
            'continue': 'space'
        }
    
    def should_trigger_pause(self, momentum_prediction, player_stamina: Dict[str, float]) -> Tuple[bool, List[str]]:
        if not self.is_enabled:
            return False, []
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_pause_time < self.pause_cooldown:
            return False, ["cooldown_active"]
        
        reasons = []
        
        # Negative momentum detection
        if (self.pause_triggers['negative_momentum'] and 
            momentum_prediction.momentum_level == 'negative' and 
            momentum_prediction.confidence > 0.75):
            reasons.append("negative_momentum_detected")
        
        # Low stamina detection
        if self.pause_triggers['low_stamina']:
            low_stamina_players = [player for player, stamina in player_stamina.items() 
                                 if stamina < 0.3]
            if len(low_stamina_players) >= 2:
                reasons.append("multiple_low_stamina")
        
        return len(reasons) > 0, reasons
    
    def execute_pause(self) -> bool:
        try:
            logger.info("Executing auto-pause...")
            pyautogui.press(self.keybinds['pause'])
            time.sleep(0.5)  # Wait for game to pause
            
            self.last_pause_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute pause: {e}")
            return False
    
    def generate_substitution_recommendations(self, 
                                            momentum_prediction, 
                                            player_stamina: Dict[str, float],
                                            current_formation: str = "4-3-3") -> List[PlayerSubstitution]:
        
        recommendations = []
        
        # Get players with low stamina
        low_stamina_players = {player: stamina for player, stamina in player_stamina.items() 
                             if stamina < 0.4}
        
        # Predefined player database (simplified)
        squad_players = {
            "ST": ["Benzema", "Morata", "Mariano"],
            "LW": ["Vinicius", "Hazard", "Rodrygo"],
            "RW": ["Asensio", "Bale", "Valverde"],
            "CM": ["Modric", "Kroos", "Casemiro", "Camavinga"],
            "LB": ["Mendy", "Marcelo", "Alaba"],
            "RB": ["Carvajal", "Vazquez", "Odriozola"],
            "CB": ["Ramos", "Varane", "Militao", "Nacho"],
            "GK": ["Courtois", "Lunin", "Areola"]
        }
        
        # Generate substitutions based on momentum and stamina
        if momentum_prediction.momentum_level == 'negative':
            # Aggressive substitutions for momentum swing
            if "Benzema" in low_stamina_players:
                recommendations.append(PlayerSubstitution(
                    player_out="Benzema",
                    player_in="Morata",
                    position="ST",
                    reason=SubstitutionReason.MOMENTUM_NEGATIVE,
                    priority=8
                ))
            
            # Fresh legs in midfield
            tired_midfielders = [p for p in low_stamina_players.keys() 
                               if p in squad_players.get("CM", [])]
            for player in tired_midfielders[:2]:  # Max 2 midfield changes
                available_subs = [p for p in squad_players["CM"] if p != player]
                if available_subs:
                    recommendations.append(PlayerSubstitution(
                        player_out=player,
                        player_in=available_subs[0],
                        position="CM",
                        reason=SubstitutionReason.STAMINA_LOW,
                        priority=6
                    ))
        
        # Sort by priority
        recommendations.sort(key=lambda x: x.priority, reverse=True)
        return recommendations[:3]  # Max 3 substitutions
    
    def generate_tactical_recommendations(self, momentum_prediction) -> TacticalAdjustment:
        if momentum_prediction.momentum_level == 'negative':
            # Defensive adjustment
            return TacticalAdjustment(
                formation_change="5-3-2",
                pressure_level="low",
                tempo="slow",
                width="narrow"
            )
        elif momentum_prediction.momentum_level == 'positive':
            # Aggressive adjustment
            return TacticalAdjustment(
                formation_change="3-4-3",
                pressure_level="high",
                tempo="fast",
                width="wide"
            )
        else:
            # Balanced approach
            return TacticalAdjustment(
                formation_change=None,
                pressure_level="medium",
                tempo="balanced",
                width="balanced"
            )
    
    def display_recommendations(self, substitutions: List[PlayerSubstitution], 
                              tactics: TacticalAdjustment):
        print("\n" + "="*50)
        print("🔄 MOMENTUM DETECTED - RECOMMENDATIONS")
        print("="*50)
        
        if substitutions:
            print("\n📋 SUBSTITUTIONS:")
            for i, sub in enumerate(substitutions, 1):
                reason_text = sub.reason.value.replace("_", " ").title()
                print(f"  {i}. {sub.player_out} → {sub.player_in} ({sub.position})")
                print(f"     Reason: {reason_text} | Priority: {sub.priority}/10")
        
        print(f"\n⚽ TACTICAL ADJUSTMENTS:")
        if tactics.formation_change:
            print(f"  Formation: {tactics.formation_change}")
        print(f"  Pressure: {tactics.pressure_level}")
        print(f"  Tempo: {tactics.tempo}")
        print(f"  Width: {tactics.width}")
        
        print("\n" + "="*50)
        print("Press SPACE to continue when ready...")
    
    def auto_execute_substitutions(self, substitutions: List[PlayerSubstitution]) -> bool:
        if not substitutions:
            return True
        
        try:
            # Open substitutions menu
            pyautogui.press(self.keybinds['substitutions'])
            time.sleep(1)
            
            # This would require more complex screen recognition
            # For now, just show the interface
            logger.info(f"Would execute {len(substitutions)} substitutions")
            
            # Close menu
            pyautogui.press('esc')
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute substitutions: {e}")
            return False
    
    def resume_game(self) -> bool:
        try:
            pyautogui.press(self.keybinds['continue'])
            time.sleep(0.5)
            logger.info("Game resumed")
            return True
        except Exception as e:
            logger.error(f"Failed to resume game: {e}")
            return False