# RL Maggi Maker

An intelligent Maggi recipe optimization system using Reinforcement Learning (SAC - Soft Actor-Critic) and XGBoost.

## Overview

This project implements a reinforcement learning agent that learns to optimize Maggi noodle recipes. The agent learns to balance various ingredients to achieve the best possible taste score while maintaining realistic proportions.

## Components

1. `sac_agent.py`: Implementation of the Soft Actor-Critic (SAC) agent and training environment
2. `testrl.py`: Testing interface for the trained agent
3. `maggi_dataset.txt`: Dataset containing Maggi recipes with taste scores
4. `maggi_xgboost_model.json`: Trained XGBoost model for taste prediction

## Key Features

- **Autonomous Learning**: Agent learns optimal ingredient ratios through exploration
- **Realistic Constraints**: Uses dataset-based bounds to ensure realistic recipes
- **Interactive Testing**: Test interface to fix one ingredient and optimize others
- **Detailed Logging**: Comprehensive training logs with performance metrics

## Architecture

### Environment
- State Space: Previous taste + normalized ingredients
- Action Space: Continuous actions for 8 ingredients
- Reward: Based on predicted taste score (using XGBoost model)

### Neural Networks
- Actor Network: Determines optimal ingredient adjustments
- Critic Networks: Estimate Q-values for actions
- Temperature Parameter: Automatically adjusted for exploration-exploitation balance

### Recipe Optimization
- Per-packet ratio analysis
- Ingredient bounds enforcement
- Taste prediction using XGBoost

## Usage

1. Training the Agent:
```python
python sac_agent.py
```

2. Testing Trained Agent:
```python
python testrl.py
```

3. Interactive Testing:
   - Fix one ingredient
   - Let the agent optimize others
   - View multiple variations and their predicted taste scores

## Requirements
- Python 3.8+
- PyTorch
- XGBoost
- NumPy
- Pandas
- Matplotlib