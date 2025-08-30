#!/usr/bin/env python3
"""
Demo script for Phase 1 of the 2D Car Racing Game
Tests the core game environment, track, cars, and visualization
"""

import sys
import os
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

def run_visual_demo():
    """Run visual demo with pygame rendering"""
    print("Starting visual demo...")
    print("Controls: ESC to quit")
    print("The RL agent (red) will take random actions")
    print("The baseline agent (blue) maintains constant speed in middle lane\n")
    
    # Create environment with rendering
    env = RaceEnvironment(render_mode="human")
    
    try:
        # Reset environment
        observation, info = env.reset()
        
        running = True
        step_count = 0
        
        while running:
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
                observation, info = env.reset()
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

def main():
    print("=== 2D Car Racing Game - Phase 1 Demo ===")
    print("DQN vs Baseline Agent\n")
    
    try:
        # Run tests
        analyze_environment()
        test_basic_functionality()
        
        # Ask user if they want to see visual demo
        while True:
            choice = input("Run visual demo? (y/n): ").lower().strip()
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
    
    print("\n=== Phase 1 Demo Completed ===")

if __name__ == "__main__":
    main()