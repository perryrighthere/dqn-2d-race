# Code Style
- Write concise Python code with concise English comment.
- Do not contain emoji in code comment.

# Workflow
- Be sure to typecheck when you're done making a series of code changes.
- No need to run code test. Just try your best to write correct code.
- After finish some code, ALWAYS tell how to run the projet.

# Project Structure
- src/environment/ - Core game environment components (track, cars, tiles, renderer)
- src/agents/ - DQN and baseline agent implementations  
- src/utils/ - Utility functions and helpers
- config/ - Configuration parameters and game settings
- demo.py - Test script with visual demonstration

# Key Components
- Track: Circular racing track with concentric lanes and lap-based racing
- Cars: BaselineCar (constant speed, middle lane) and RLCar (full physics, lane changes)
- Special Tiles: Acceleration/deceleration tiles in outer/inner lanes only
- Environment: Gymnasium-compatible RL environment with 5 discrete actions
- DQN Agent: Double DQN with dueling architecture for competitive racing
- Renderer: Real-time circular track visualization with pygame

# Development Commands
- Install dependencies: pip install -r requirements.txt
- Run demo: python demo.py
- Run baseline benchmarks: python baseline_benchmark.py
- Train DQN agent (quick): python train_dqn.py --config fast
- Train DQN agent (standard): python train_dqn.py --config standard
- Train DQN agent (custom): python train_dqn.py --episodes 1500 --render
- Type check: python -m py_compile src/environment/*.py src/agents/*.py config/*.py
- Test environment: python -c "from src.environment.race_environment import RaceEnvironment; env = RaceEnvironment(); print('Environment loads successfully')"