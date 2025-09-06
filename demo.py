#!/usr/bin/env python3
"""
Demo script for Phase 1 of the 2D Car Racing Game
Tests the core game environment, track, cars, and visualization
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.environment.race_environment import RaceEnvironment
import numpy as np
import time

def test_basic_functionality():
    """Test basic functionality without rendering"""
    print("Testing basic environment functionality...")
    
    # Create environment without rendering
    env = RaceEnvironment(render_mode=None)
    
    # Reset environment
    observation, info = env.reset()
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial info: {info}")
    
    # Run a few steps with random actions
    for step in range(10):
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"Step {step+1}: Action={action}, Reward={reward:.2f}, "
              f"RL Laps={info['rl_car_laps']}, RL Angle={info['rl_car_angle']:.2f}, "
              f"Baseline Laps={info['baseline_car_laps']}, Baseline Angle={info['baseline_car_angle']:.2f}")
        
        if terminated:
            print(f"Race finished! Winner: {info['winner']}")
            break
    
    env.close()
    print("Basic functionality test completed!\n")

def run_visual_demo(use_trained_model=False, model_path=None, tile_density=None, accel_ratio=None, seed=None):
    """Run visual demo with pygame rendering"""
    print("Starting visual demo...")
    print("Controls: ESC to quit")
    
    agent = None
    if use_trained_model and model_path and os.path.exists(model_path):
        try:
            from src.agents.dqn_agent import create_racing_dqn_agent
            config = {
                'state_size': 11,
                'action_size': 5,
                'epsilon': 0.0,
                'learning_rate': 0.0005
            }
            agent = create_racing_dqn_agent(config=config)
            agent.load_model(model_path, load_optimizer=False)
            print(f"Using trained DQN model: {model_path}")
            print("The RL agent (red) will use trained DQN policy")
        except Exception as e:
            print(f"Failed to load trained model: {e}")
            print("Falling back to random actions")
            agent = None
    
    if agent is None:
        print("The RL agent (red) will take random actions")
    
    print("The baseline agent (blue) maintains constant speed in middle lane\n")
    
    # Create environment with rendering
    env = RaceEnvironment(render_mode="human")
    
    try:
        # Reset environment
        opts = {'tile_density': tile_density, 'accel_ratio': accel_ratio}
        observation, info = env.reset(seed=seed, options=opts)
        
        running = True
        step_count = 0
        
        while running:
            if agent is not None:
                # Use trained agent
                action = agent.act(observation, training=False)
            else:
                # Take random action for demo purposes
                action = env.action_space.sample()
                
                # Occasionally bias towards lane changes for more interesting demo
                if np.random.random() < 0.3:
                    action = np.random.choice([1, 2])  # Prefer lane changes
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Print periodic updates
            if step_count % 60 == 0:  # Every second at 60 FPS
                print(f"Step {step_count}: RL Laps={info['rl_car_laps']}, "
                      f"Baseline Laps={info['baseline_car_laps']}")
            
            # Check if race finished
            if terminated:
                print(f"\nRace finished after {step_count} steps!")
                print(f"Winner: {info['winner']}")
                print(f"Race time: {info['race_time']:.1f} seconds")
                print("Press ESC to exit or any key to restart...")
                
                # Wait for user input
                time.sleep(2)
                observation, info = env.reset(seed=seed, options=opts)
                step_count = 0
                
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        
    finally:
        env.close()
        print("Visual demo completed!")

def analyze_environment():
    """Analyze environment properties"""
    print("Analyzing environment properties...")
    
    env = RaceEnvironment()
    
    print(f"Action space: {env.action_space}")
    print(f"Action meanings: {env.get_action_meanings()}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset and analyze initial state
    obs, info = env.reset()
    print(f"Observation dimensions: {len(obs)}")
    print(f"Track radius: {env.track.radius}")
    print(f"Track circumference: {env.track.circumference:.1f}")
    print(f"Number of lanes: {env.track.num_lanes}")
    print(f"Middle lane ID: {env.track.middle_lane_id}")
    print(f"Laps to win: {env.track.laps_to_win}")
    
    # Analyze tile distribution
    tile_counts = env.tile_manager.get_tile_count()
    print(f"Special tiles generated: {tile_counts}")
    
    total_tiles = sum(tile_counts.values())
    print(f"Total special tiles: {total_tiles}")
    
    env.close()
    print("Environment analysis completed!\n")

def run_headless_demo(model_path=None, races=10, tile_density=None, accel_ratio=None, seed=None):
    from src.agents.dqn_agent import create_racing_dqn_agent
    agent = None
    if model_path and os.path.exists(model_path):
        config = {
            'state_size': 11,
            'action_size': 5,
            'epsilon': 0.0,
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'buffer_size': 100000,
            'batch_size': 64,
            'network_type': 'double'
        }
        agent = create_racing_dqn_agent(config=config)
        agent.load_model(model_path, load_optimizer=False)
        print(f"Loaded model: {model_path}")
    else:
        print("No model provided; using random actions")

    wins = 0
    rl_times = []
    for i in range(races):
        env = RaceEnvironment(render_mode=None)
        obs, info = env.reset(seed=(seed or 0) + i, options={'tile_density': tile_density, 'accel_ratio': accel_ratio})
        while True:
            if agent is not None:
                action = agent.act(obs, training=False)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        if info.get('winner') == 'RL':
            wins += 1
        rl_times.append(info.get('race_time', 0.0))
        env.close()

    print("\n=== Headless Demo Summary ===")
    print(f"Races: {races}")
    print(f"Win rate: {wins/races:.1%}")
    if rl_times:
        print(f"Avg RL race time: {sum(rl_times)/len(rl_times):.2f}s")

def main():
    parser = argparse.ArgumentParser(description='2D Car Racing Demo (visual or headless)')
    parser.add_argument('--model', '-m', default=None, help='Path to trained model')
    parser.add_argument('--headless', action='store_true', help='Run without visualization')
    parser.add_argument('--races', type=int, default=3, help='Number of races (headless only)')
    parser.add_argument('--tile-density', type=float, default=None, help='Tile density (e.g., 0.5, 0.8, 1.1)')
    parser.add_argument('--accel-ratio', type=float, default=None, help='Acceleration tile ratio (0..1)')
    parser.add_argument('--seed', type=int, default=None, help='Base seed')
    args = parser.parse_args()

    print("=== 2D Car Racing Game Demo ===")
    print("DQN vs Baseline Agent\n")

    # Non-interactive CLI modes
    if args.headless or args.model or args.tile_density is not None or args.accel_ratio is not None:
        if args.headless:
            run_headless_demo(
                model_path=args.model,
                races=args.races,
                tile_density=args.tile_density,
                accel_ratio=args.accel_ratio,
                seed=args.seed,
            )
        else:
            use_trained = bool(args.model)
            run_visual_demo(
                use_trained_model=use_trained,
                model_path=args.model,
                tile_density=args.tile_density,
                accel_ratio=args.accel_ratio,
                seed=args.seed,
            )
        print("\n=== Demo Completed ===")
        return

    # Default interactive path (unchanged behavior)
    try:
        analyze_environment()
        test_basic_functionality()
        model_path = "models/dqn_racing_final.pth"
        has_trained_model = os.path.exists(model_path)
        if has_trained_model:
            print(f"Found trained model: {model_path}")
        while True:
            if has_trained_model:
                print("Demo options:")
                print("  1. Random agent vs Baseline")
                print("  2. Trained DQN vs Baseline")
                print("  3. Skip demo")
                try:
                    choice = input("Choose option (1/2/3): ").strip()
                except EOFError:
                    print("Running demo with enhanced visuals...")
                    choice = "1"
                if choice == "1":
                    run_visual_demo(use_trained_model=False)
                    break
                elif choice == "2":
                    run_visual_demo(use_trained_model=True, model_path=model_path)
                    break
                elif choice == "3":
                    print("Skipping visual demo.")
                    break
                else:
                    print("Please enter 1, 2, or 3")
            else:
                try:
                    choice = input("Run visual demo? (y/n): ").lower().strip()
                except EOFError:
                    print("Running demo with enhanced visuals...")
                    choice = "y"
                if choice in ['y', 'yes']:
                    run_visual_demo()
                    break
                elif choice in ['n', 'no']:
                    print("Skipping visual demo.")
                    break
                else:
                    print("Please enter 'y' or 'n'")
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    print("\n=== Demo Completed ===")

if __name__ == "__main__":
    main()
