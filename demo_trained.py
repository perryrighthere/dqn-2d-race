#!/usr/bin/env python3
"""
Demo script for trained DQN agent vs Baseline agent
Shows the trained DQN model competing against the baseline agent
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.environment.race_environment import RaceEnvironment
from src.agents.dqn_agent import create_racing_dqn_agent
import numpy as np
import time
import torch

def load_trained_agent(model_path: str):
    """Load a trained DQN agent from saved model
    
    Args:
        model_path: Path to saved model file
        
    Returns:
        Loaded DQN agent
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create agent with same configuration as training
    config = {
        'state_size': 11,
        'action_size': 5,
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon': 0.0,  # No exploration for evaluation
        'epsilon_min': 0.0,
        'epsilon_decay': 1.0,
        'buffer_size': 100000,
        'batch_size': 64,
        'target_update_freq': 1000,
        'network_type': 'double'
    }
    
    agent = create_racing_dqn_agent(config=config)
    
    # Load the trained model
    agent.load_model(model_path, load_optimizer=False)
    print(f"Loaded trained model from: {model_path}")
    
    return agent

def run_trained_demo(model_path: str, num_races: int = 3):
    """Run demo with trained DQN agent
    
    Args:
        model_path: Path to trained model
        num_races: Number of races to run
    """
    print("=== Trained DQN vs Baseline Demo ===")
    print(f"Model: {model_path}")
    print(f"Running {num_races} races...")
    print("Controls: ESC to quit")
    print("Red car: Trained DQN Agent")
    print("Blue car: Baseline Agent\n")
    
    # Load trained agent
    try:
        agent = load_trained_agent(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment with rendering
    env = RaceEnvironment(render_mode="human")
    
    wins = {"dqn": 0, "baseline": 0, "timeout": 0}
    race_times = {"dqn": [], "baseline": []}
    
    try:
        for race in range(num_races):
            print(f"\n=== Race {race + 1}/{num_races} ===")
            
            # Reset environment
            observation, info = env.reset()
            
            race_start_time = time.time()
            step_count = 0
            
            while True:
                # Get action from trained agent (no exploration)
                action = agent.act(observation, training=False)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                observation = obs
                step_count += 1
                
                # Print periodic updates
                if step_count % 120 == 0:  # Every 2 seconds at 60 FPS
                    print(f"  Step {step_count}: DQN Laps={info['rl_car_laps']}, "
                          f"Baseline Laps={info['baseline_car_laps']}")
                
                # Check if race finished
                if terminated:
                    race_time = time.time() - race_start_time
                    winner = info.get('winner', 'timeout')
                    if winner is None:
                        winner = 'timeout'
                    
                    print(f"  Race {race + 1} finished after {step_count} steps ({race_time:.1f}s)")
                    print(f"  Winner: {winner}")
                    print(f"  Race time: {info.get('race_time', 0):.1f} seconds")
                    
                    # Track statistics
                    if winner not in wins:
                        wins[winner] = 0
                    wins[winner] += 1
                    if winner in race_times:
                        race_times[winner].append(info.get('race_time', 0))
                    
                    # Wait before next race
                    if race < num_races - 1:
                        print("  Next race in 3 seconds...")
                        time.sleep(3)
                    
                    break
                
                # Handle timeout
                if truncated:
                    print(f"  Race {race + 1} timed out after {step_count} steps")
                    wins["timeout"] += 1
                    break
                    
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
        
    finally:
        env.close()
    
    # Print final statistics
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    print(f"Total races: {num_races}")
    print(f"DQN wins: {wins['dqn']} ({wins['dqn']/num_races*100:.1f}%)")
    print(f"Baseline wins: {wins['baseline']} ({wins['baseline']/num_races*100:.1f}%)")
    print(f"Timeouts: {wins['timeout']} ({wins['timeout']/num_races*100:.1f}%)")
    
    if race_times['dqn']:
        avg_dqn_time = np.mean(race_times['dqn'])
        print(f"Average DQN race time: {avg_dqn_time:.1f}s")
    
    if race_times['baseline']:
        avg_baseline_time = np.mean(race_times['baseline'])
        print(f"Average Baseline race time: {avg_baseline_time:.1f}s")
        
        if race_times['dqn']:
            improvement = avg_baseline_time - avg_dqn_time
            print(f"Time improvement: {improvement:+.1f}s")

def run_headless_evaluation(model_path: str, num_races: int = 10):
    """Run headless evaluation for quick performance testing
    
    Args:
        model_path: Path to trained model
        num_races: Number of races to run
    """
    print("=== Headless Evaluation ===")
    print(f"Model: {model_path}")
    print(f"Running {num_races} races without visualization...\n")
    
    # Load trained agent
    try:
        agent = load_trained_agent(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Create environment without rendering
    env = RaceEnvironment(render_mode=None)
    
    wins = {"dqn": 0, "baseline": 0, "timeout": 0}
    race_times = {"dqn": [], "baseline": []}
    rewards = []
    
    for race in range(num_races):
        # Reset environment
        observation, info = env.reset()
        total_reward = 0
        
        while True:
            # Get action from trained agent
            action = agent.act(observation, training=False)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            observation = obs
            total_reward += reward
            
            if terminated:
                winner = info.get('winner', 'timeout')
                if winner is None:
                    winner = 'timeout'
                
                if winner not in wins:
                    wins[winner] = 0
                wins[winner] += 1
                
                if winner in race_times:
                    race_times[winner].append(info.get('race_time', 0))
                
                rewards.append(total_reward)
                race_time = info.get('race_time', 0)
                print(f"Race {race + 1}: Winner={winner}, Time={race_time:.1f}s, Reward={total_reward:.1f}")
                break
                
            if truncated:
                wins["timeout"] += 1
                rewards.append(total_reward)
                print(f"Race {race + 1}: TIMEOUT, Reward={total_reward:.1f}")
                break
    
    env.close()
    
    # Print statistics
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"Win rate: {wins['dqn']/num_races*100:.1f}%")
    print(f"Average reward: {np.mean(rewards):.1f}")
    
    if race_times['dqn'] and race_times['baseline']:
        avg_dqn = np.mean(race_times['dqn'])
        avg_baseline = np.mean(race_times['baseline'])
        improvement = avg_baseline - avg_dqn
        print(f"Average race time improvement: {improvement:+.1f}s vs baseline")

def main():
    parser = argparse.ArgumentParser(description="Demo trained DQN agent")
    parser.add_argument("--model", "-m", 
                       default="models/dqn_racing_final.pth",
                       help="Path to trained model file")
    parser.add_argument("--races", "-r", type=int, default=3,
                       help="Number of races to run")
    parser.add_argument("--headless", action="store_true",
                       help="Run without visualization for quick evaluation")
    
    args = parser.parse_args()
    
    try:
        if args.headless:
            run_headless_evaluation(args.model, args.races)
        else:
            run_trained_demo(args.model, args.races)
            
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()