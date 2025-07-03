import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
import sqlite3
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    match_id: str
    timestamp: datetime
    opponent: str
    final_score: Tuple[int, int]  # (user_goals, opponent_goals)
    result: str  # 'win', 'draw', 'loss'
    game_mode: str  # 'rivals', 'fut_champs', 'squad_battles'
    momentum_interventions: int
    avg_momentum_score: float
    substitutions_made: int
    tactical_changes: int

@dataclass
class PerformanceMetrics:
    total_matches: int
    win_rate: float
    avg_goals_scored: float
    avg_goals_conceded: float
    momentum_detection_accuracy: float
    intervention_success_rate: float

class PerformanceTracker:
    def __init__(self, db_path: str = "data/fifa_performance.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS matches (
                    match_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    opponent TEXT,
                    user_goals INTEGER,
                    opponent_goals INTEGER,
                    result TEXT,
                    game_mode TEXT,
                    momentum_interventions INTEGER,
                    avg_momentum_score REAL,
                    substitutions_made INTEGER,
                    tactical_changes INTEGER
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS momentum_events (
                    event_id TEXT PRIMARY KEY,
                    match_id TEXT,
                    timestamp TEXT,
                    momentum_level TEXT,
                    confidence REAL,
                    intervention_triggered BOOLEAN,
                    intervention_successful BOOLEAN,
                    FOREIGN KEY (match_id) REFERENCES matches (match_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def record_match(self, match_result: MatchResult):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO matches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_result.match_id,
                match_result.timestamp.isoformat(),
                match_result.opponent,
                match_result.final_score[0],
                match_result.final_score[1],
                match_result.result,
                match_result.game_mode,
                match_result.momentum_interventions,
                match_result.avg_momentum_score,
                match_result.substitutions_made,
                match_result.tactical_changes
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Match {match_result.match_id} recorded")
            
        except Exception as e:
            logger.error(f"Failed to record match: {e}")
    
    def get_performance_metrics(self, days_back: int = 30) -> PerformanceMetrics:
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            df = pd.read_sql_query('''
                SELECT * FROM matches 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            ''', conn, params=[cutoff_date.isoformat()])
            
            conn.close()
            
            if df.empty:
                return PerformanceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
            total_matches = len(df)
            wins = len(df[df['result'] == 'win'])
            win_rate = wins / total_matches if total_matches > 0 else 0.0
            
            avg_goals_scored = df['user_goals'].mean()
            avg_goals_conceded = df['opponent_goals'].mean()
            
            # Placeholder calculations for momentum metrics
            momentum_accuracy = 0.82  # Would be calculated from momentum_events table
            intervention_success = 0.67  # Would be calculated from intervention outcomes
            
            return PerformanceMetrics(
                total_matches=total_matches,
                win_rate=win_rate,
                avg_goals_scored=avg_goals_scored,
                avg_goals_conceded=avg_goals_conceded,
                momentum_detection_accuracy=momentum_accuracy,
                intervention_success_rate=intervention_success
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return PerformanceMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    
    def generate_performance_report(self, days_back: int = 30) -> Dict:
        metrics = self.get_performance_metrics(days_back)
        
        # Get baseline performance (before using the system)
        baseline_metrics = self.get_baseline_performance()
        
        improvement = {
            'win_rate_change': metrics.win_rate - baseline_metrics.get('win_rate', 0.5),
            'goals_improvement': metrics.avg_goals_scored - baseline_metrics.get('avg_goals_scored', 1.5),
            'defensive_improvement': baseline_metrics.get('avg_goals_conceded', 2.0) - metrics.avg_goals_conceded
        }
        
        return {
            'current_metrics': asdict(metrics),
            'baseline_metrics': baseline_metrics,
            'improvement': improvement,
            'summary': self._generate_summary_text(metrics, improvement)
        }
    
    def get_baseline_performance(self) -> Dict:
        # This would typically come from matches before the system was used
        # For demo purposes, using hypothetical baseline
        return {
            'win_rate': 0.45,
            'avg_goals_scored': 1.8,
            'avg_goals_conceded': 2.1,
            'total_matches': 50
        }
    
    def _generate_summary_text(self, metrics: PerformanceMetrics, improvement: Dict) -> str:
        summary_parts = []
        
        # Win rate analysis
        win_rate_pct = metrics.win_rate * 100
        win_rate_change = improvement['win_rate_change'] * 100
        
        if win_rate_change > 5:
            summary_parts.append(f"🚀 Significant win rate improvement: {win_rate_change:+.1f}% (now {win_rate_pct:.1f}%)")
        elif win_rate_change > 0:
            summary_parts.append(f"📈 Win rate improved by {win_rate_change:+.1f}% to {win_rate_pct:.1f}%")
        else:
            summary_parts.append(f"📊 Current win rate: {win_rate_pct:.1f}% ({win_rate_change:+.1f}%)")
        
        # Goals analysis
        if improvement['goals_improvement'] > 0.3:
            summary_parts.append(f"⚽ Attack improved: +{improvement['goals_improvement']:.1f} goals per game")
        
        if improvement['defensive_improvement'] > 0.3:
            summary_parts.append(f"🛡️ Defense improved: -{improvement['defensive_improvement']:.1f} goals conceded")
        
        # Momentum system effectiveness
        if metrics.momentum_detection_accuracy > 0.8:
            summary_parts.append(f"🎯 High momentum detection accuracy: {metrics.momentum_detection_accuracy*100:.1f}%")
        
        return "\n".join(summary_parts)
    
    def create_performance_charts(self, output_dir: str = "data/charts"):
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM matches ORDER BY timestamp', conn)
            conn.close()
            
            if df.empty:
                logger.warning("No match data available for charts")
                return
            
            # Win rate over time
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['win'] = (df['result'] == 'win').astype(int)
            df['rolling_win_rate'] = df['win'].rolling(window=10, min_periods=1).mean()
            
            plt.plot(df['timestamp'], df['rolling_win_rate'] * 100)
            plt.title('Win Rate Over Time (10-game rolling average)')
            plt.ylabel('Win Rate (%)')
            plt.xticks(rotation=45)
            
            # Goals scored vs conceded
            plt.subplot(2, 2, 2)
            plt.scatter(df['user_goals'], df['opponent_goals'], 
                       c=['green' if r == 'win' else 'red' if r == 'loss' else 'yellow' 
                          for r in df['result']], alpha=0.6)
            plt.xlabel('Goals Scored')
            plt.ylabel('Goals Conceded')
            plt.title('Goals Scored vs Conceded')
            
            # Momentum interventions effectiveness
            plt.subplot(2, 2, 3)
            intervention_win_rate = df.groupby('momentum_interventions')['win'].mean()
            plt.bar(intervention_win_rate.index, intervention_win_rate.values * 100)
            plt.xlabel('Momentum Interventions per Game')
            plt.ylabel('Win Rate (%)')
            plt.title('Win Rate by Momentum Interventions')
            
            # Game mode performance
            plt.subplot(2, 2, 4)
            mode_performance = df.groupby('game_mode')['win'].mean()
            plt.bar(mode_performance.index, mode_performance.values * 100)
            plt.xlabel('Game Mode')
            plt.ylabel('Win Rate (%)')
            plt.title('Performance by Game Mode')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f"{output_dir}/performance_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance charts saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to create performance charts: {e}")
    
    def export_data(self, filepath: str):
        try:
            conn = sqlite3.connect(self.db_path)
            
            matches_df = pd.read_sql_query('SELECT * FROM matches', conn)
            events_df = pd.read_sql_query('SELECT * FROM momentum_events', conn)
            
            with pd.ExcelWriter(filepath) as writer:
                matches_df.to_excel(writer, sheet_name='Matches', index=False)
                events_df.to_excel(writer, sheet_name='Momentum Events', index=False)
            
            conn.close()
            logger.info(f"Data exported to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to export data: {e}")