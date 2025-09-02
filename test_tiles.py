#!/usr/bin/env python3
"""
Test script to visualize the enhanced tile system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.environment.race_environment import RaceEnvironment
import time

def test_tile_generation():
    """Test tile generation with different seeds"""
    print("=== Testing Enhanced Tile System ===\n")
    
    for seed in [42, 123, 456]:
        print(f"Seed {seed}:")
        env = RaceEnvironment()
        env.reset(seed=seed)
        
        stats = env.tile_manager.get_detailed_tile_stats()
        print(f"  Total tiles: {stats['total_tiles']}")
        print(f"  Tiles per lane: {stats['average_tiles_per_lane']:.1f}")
        print(f"  Acceleration tiles: {stats['tiles_by_type']['acceleration']}")
        print(f"  Deceleration tiles: {stats['tiles_by_type']['deceleration']}")
        print()
        
        env.close()

def run_visual_tile_test():
    """Run visual demo to see the tiles in action"""
    print("Starting visual tile test...")
    print("Look for the + (acceleration) and - (deceleration) symbols")
    print("You should see many more tiles than before!")
    print("Controls: ESC to quit\n")
    
    env = RaceEnvironment(render_mode="human")
    
    try:
        # Reset with a good seed
        observation, info = env.reset(seed=42)
        
        # Show tile stats
        stats = env.tile_manager.get_detailed_tile_stats()
        print(f"Displaying {stats['total_tiles']} tiles:")
        for lane, data in stats['tiles_by_lane'].items():
            print(f"  {lane}: {data['total']} tiles")
        
        step_count = 0
        tile_interactions = 0
        
        while step_count < 1800:  # Run for 30 seconds
            # Take some interesting actions to show tile interactions
            if step_count < 300:
                action = 1  # Go to inner lane
            elif step_count < 600:
                action = 2  # Go to outer lane
            elif step_count < 900:
                action = 1  # Back to inner lane
            else:
                action = 0  # Stay in current lane
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            
            # Count tile interactions (positive rewards indicate tile hits)
            if reward > 0.5:  # Tile interaction reward threshold
                tile_interactions += 1
            
            # Print periodic updates
            if step_count % 300 == 0:  # Every 5 seconds
                print(f"Step {step_count}: Tile interactions so far: {tile_interactions}")
            
            if terminated or truncated:
                print(f"Race finished! Total tile interactions: {tile_interactions}")
                break
                
        print(f"Test completed after {step_count} steps with {tile_interactions} tile interactions")
        
    except KeyboardInterrupt:
        print("Test interrupted by user")
        
    finally:
        env.close()

def main():
    print("Enhanced Tile System Test\n")
    
    # Test tile generation
    test_tile_generation()
    
    # Ask user if they want visual test
    choice = input("Run visual test to see tiles in action? (y/n): ").lower().strip()
    if choice in ['y', 'yes']:
        run_visual_tile_test()
    else:
        print("Skipping visual test.")
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    main()