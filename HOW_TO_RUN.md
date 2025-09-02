# How to Run the DQN 2D Car Racing Project

## 📋 Prerequisites

- Python 3.8+ (tested with Python 3.13)
- Virtual environment (recommended)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test the Environment
```bash
python demo.py
```

### 3. Train Your DQN Agent
```bash
# Quick training (200 episodes)
python train_dqn.py --config fast --episodes 200

# Standard training (1000 episodes)  
python train_dqn.py --config standard --episodes 1000
```

### 4. Test Trained Agent
```bash
# Visual demo with trained model
python demo_trained.py

# Quick performance evaluation
python demo_trained.py --headless --races 20
```

---

## 🎮 Running Different Components

### Environment Testing & Demo

#### Basic Demo (Random Agent)
```bash
python demo.py
```
- Tests environment functionality
- Shows random agent vs baseline
- Interactive choice for visual demo

#### Advanced Demo (Trained Agent)
```bash
python demo_trained.py
```
- Shows trained DQN vs baseline
- Real-time racing statistics
- Win rate and performance analysis

### Training the DQN Agent

#### Quick Training (Fast Testing)
```bash
python train_dqn.py --config fast --episodes 200
```
- 200 episodes (~1 minute)
- Good for testing setup
- Saves models every 50 episodes

#### Standard Training (Balanced)
```bash
python train_dqn.py --config standard --episodes 1000
```
- 1000 episodes (~5-10 minutes)
- Good balance of time and performance
- Saves models every 100 episodes

#### Deep Training (Best Performance)
```bash
python train_dqn.py --config deep --episodes 2000
```
- 2000+ episodes (~15-20 minutes)
- Best chance for high win rate
- More thorough learning

#### Custom Training
```bash
# Custom configuration
python train_dqn.py --episodes 1500 --render --save-dir custom_models

# Training with visualization (slower)
python train_dqn.py --config standard --episodes 500 --render
```

### Baseline Performance Testing

#### Run Baseline Benchmarks
```bash
python baseline_benchmark.py
```
- Tests baseline agent consistency
- Generates performance statistics
- Creates benchmark reports

### Model Evaluation & Comparison

#### Visual Evaluation
```bash
# Use latest trained model
python demo_trained.py --model models/dqn_racing_final.pth --races 5

# Test specific checkpoint
python demo_trained.py --model models/dqn_racing_episode_100.pth --races 3
```

#### Headless Performance Testing
```bash
# Quick evaluation (no graphics)
python demo_trained.py --headless --races 20

# Comprehensive evaluation
python demo_trained.py --headless --races 100
```

---

## 📊 Understanding the Output

### Training Output
```
Episode  100/1000 | Reward:  481.92 | Win Rate: 15.0% | Epsilon: 0.750 | Time: 18s
```
- **Episode**: Current training episode
- **Reward**: Average reward over last 10 episodes
- **Win Rate**: Percentage of recent wins vs baseline
- **Epsilon**: Current exploration rate
- **Time**: Total training time

### Demo Output
```
Race 1: Winner=dqn, Time=165.2s, Reward=523.4
```
- **Winner**: Which agent won (dqn/baseline/timeout)
- **Time**: Race completion time in seconds
- **Reward**: Total reward earned by DQN

### Performance Targets
- **Baseline Performance**: 180.00 seconds (consistent)
- **Target Win Rate**: 60%+ wins against baseline
- **Target Improvement**: 5-15 second faster than baseline

---

## 📁 Generated Files

### Models
- `models/dqn_racing_final.pth` - Final trained model
- `models/dqn_racing_episode_X.pth` - Checkpoint models

### Logs
- `logs/training_history_*.json` - Training statistics
- `logs/training_plots_*.png` - Training progress plots

### Benchmarks
- `baseline_benchmark_*.json` - Baseline performance data
- `PHASE*_COMPLETED.md` - Phase completion reports

---

## ⚙️ Configuration Options

### Training Configurations

#### Fast Config
- Episodes: 200
- Learning Rate: 0.001
- Higher exploration decay
- Quick testing/validation

#### Standard Config  
- Episodes: 1000
- Learning Rate: 0.0005
- Balanced parameters
- Good general performance

#### Deep Config
- Episodes: 2000+
- Learning Rate: 0.0003
- Slower, thorough learning
- Best final performance

### Command Line Options

#### Training Script (`train_dqn.py`)
```bash
--config {fast,standard,deep}  # Preset configurations
--episodes N                   # Number of training episodes
--render                      # Show training visualization
--save-dir DIR               # Directory to save models
--seed N                     # Random seed for reproducibility
```

#### Demo Script (`demo_trained.py`)
```bash
--model PATH                 # Path to trained model
--races N                    # Number of races to run
--headless                   # Run without visualization
```

---

## 🔧 Troubleshooting

### Common Issues

#### Import Errors
```bash
# Make sure you're in the project directory
cd /path/to/dqn-2d-race

# Install dependencies
pip install -r requirements.txt
```

#### Model Loading Errors
```bash
# Check if model file exists
ls models/

# Use specific model path
python demo_trained.py --model models/dqn_racing_episode_100.pth
```

#### Training Performance Issues
```bash
# Start with fast training to test
python train_dqn.py --config fast --episodes 100

# Check system resources (training can be CPU intensive)
# Consider reducing episodes or using headless mode
```

### Environment Validation
```bash
# Test environment loading
python -c "from src.environment.race_environment import RaceEnvironment; env = RaceEnvironment(); print('Environment loads successfully')"

# Test DQN agent
python -c "from src.agents.dqn_agent import DQNAgent; print('DQN agent imports successfully')"

# Type check all files
python -m py_compile src/environment/*.py src/agents/*.py config/*.py
```

---

## 🏁 Racing Game Controls

### Visual Demo Controls
- **ESC**: Quit demo
- **Window Close**: Exit application

### Racing Mechanics
- **Red Car**: DQN Agent (your trained AI)
- **Blue Car**: Baseline Agent (constant speed, middle lane)
- **Special Tiles**: 
  - **+ symbols**: Acceleration tiles (speed boost)
  - **- symbols**: Deceleration tiles (speed reduction)
- **Race Objective**: Complete 3 laps first to win

---

## 📈 Performance Analysis

### Expected Training Progression
1. **Episodes 0-200**: Random exploration, low win rate (<10%)
2. **Episodes 200-500**: Learning basic racing, improving win rate (10-30%)
3. **Episodes 500-1000**: Strategic behavior, good win rate (30-60%)
4. **Episodes 1000+**: Refined strategy, high win rate (60%+)

### Success Metrics
- **Win Rate**: >60% against baseline agent
- **Time Improvement**: 5-15 seconds faster than 180s baseline
- **Consistency**: Stable performance across multiple races
- **Strategy**: Evidence of lane-changing and tile utilization

---

## 🎯 Next Steps

After successful training:

1. **Evaluate Performance**: Use `demo_trained.py` to test win rate
2. **Experiment with Training**: Try different configurations
3. **Analyze Strategy**: Watch races to see learned behaviors
4. **Optimize Further**: Adjust hyperparameters for better results

---

## 📞 Support

For issues or questions:
- Check the troubleshooting section above
- Review the phase completion documents (`PHASE*_COMPLETED.md`)
- Examine the project plan (`PROJECT_PLAN.md`)

The project implements a complete reinforcement learning pipeline for autonomous racing, from environment setup through DQN training to competitive evaluation.