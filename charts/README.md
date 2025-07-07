# Performance Charts

This directory contains automatically generated performance visualizations:

- `performance_dashboard.png` - Main dashboard with win rates, goals, and game mode analysis
- `model_training_progress.png` - ML model training accuracy over epochs

Charts are updated automatically when running:
```bash
python main.py --report
```

## Chart Types

### Performance Dashboard
- Win rate trends over time
- Goals scored vs conceded scatter plot  
- Win rate by momentum interventions
- Performance breakdown by game mode

### Training Progress
- Model accuracy progression during training
- Training vs validation curves
- Convergence analysis