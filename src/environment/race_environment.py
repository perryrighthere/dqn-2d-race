import gymnasium as gym
import numpy as np
import math
from typing import List, Tuple, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.game_config import *
from .track import Track
from .car import Car, BaselineCar, RLCar
from .special_tiles import TileManager
from .renderer import GameRenderer

class RaceEnvironment(gym.Env):
    """
    Circular 2D Car Racing Environment for RL Agent vs Baseline Agent
    """
    
    def __init__(self, render_mode: str = None):
        super().__init__()
        
        # Environment configuration
        self.render_mode = render_mode
        
        # Initialize components
        self.track = Track()
        self.tile_manager = TileManager(self.track)
        
        # Initialize cars
        self.baseline_car = BaselineCar(angle=0, track=self.track)
        self.rl_car = RLCar(angle=0, lane=0)  # Start in innermost lane
        
        self.cars = [self.baseline_car, self.rl_car]
        
        # Renderer for visualization
        self.renderer = None
        if render_mode == "human":
            self.renderer = GameRenderer()
            
        # Environment state
        self.race_time = 0.0
        self.episode = 0
        self.race_finished = False
        self.winner = None
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(5)  # [stay, left, right, accelerate, decelerate]
        
        # Observation space: car state + tile information for each lane
        obs_size = 5 + (self.track.num_lanes * 2)  # 5 car states + 2 per lane for tiles
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=2.0, shape=(obs_size,), dtype=np.float32
        )
        
    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            
        # Reset track and generate new tiles
        self.track.reset()
        self.tile_manager.reset(seed=seed)
        
        # Reset cars to starting positions (angle 0, different lanes)
        self.baseline_car.reset(angle=0, lane=self.track.middle_lane_id)
        self.baseline_car.update_radius_from_lane(self.track)
        
        self.rl_car.reset(angle=0, lane=0)  # Start in innermost lane
        self.rl_car.update_radius_from_lane(self.track)
        
        # Reset environment state
        self.race_time = 0.0
        self.episode += 1
        self.race_finished = False
        self.winner = None
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step of the environment"""
        
        # Apply action to RL car
        self._apply_action(action)
        
        # Update car physics
        for car in self.cars:
            car.update_physics(track=self.track)
            
        # Check tile collisions
        for car in self.cars:
            self.tile_manager.check_car_collisions(car)
            
        # Update race time
        self.race_time += 1.0 / FPS
        
        # Check if race is finished
        terminated, winner = self._check_race_finished()
        
        # Calculate reward
        reward = self._calculate_reward(terminated, winner)
        
        # Get observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        # Render if needed
        if self.render_mode == "human" and self.renderer:
            if not self.renderer.handle_events():
                terminated = True
            self.renderer.render(
                self.track, self.cars, self.tile_manager, 
                self.race_time, self.episode
            )
            
        return observation, reward, terminated, False, info
        
    def _apply_action(self, action: int):
        """Apply action to the RL car"""
        # Action mapping: 0=stay, 1=inner_lane, 2=outer_lane, 3=accelerate, 4=decelerate
        
        if action == 1:  # Move to inner lane (smaller radius)
            valid_lanes = self.track.get_valid_lane_changes(self.rl_car.current_lane)
            if self.rl_car.current_lane > 0:
                new_lane = self.rl_car.current_lane - 1
                if new_lane in valid_lanes:
                    self.rl_car.change_lane(new_lane)
                    
        elif action == 2:  # Move to outer lane (larger radius)
            valid_lanes = self.track.get_valid_lane_changes(self.rl_car.current_lane)
            if self.rl_car.current_lane < self.track.num_lanes - 1:
                new_lane = self.rl_car.current_lane + 1
                if new_lane in valid_lanes:
                    self.rl_car.change_lane(new_lane)
                    
        elif action == 3:  # Accelerate
            self.rl_car.apply_acceleration(1.0)
            
        elif action == 4:  # Decelerate
            self.rl_car.apply_acceleration(-1.0)
            
        # Action 0 (stay) requires no changes
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation for RL agent"""
        # Get RL car state
        rl_state = self.rl_car.get_state(self.track)
        
        # Get tile information
        tile_state = self.tile_manager.get_state_representation(self.rl_car)
        
        # Combine states
        observation = np.concatenate([rl_state, tile_state])
        
        return observation.astype(np.float32)
        
    def _calculate_reward(self, terminated: bool, winner: str) -> float:
        """Calculate reward for the current step"""
        reward = 0.0
        
        # Base reward: small positive for making progress
        reward += 0.1
        
        # Reward for being ahead of baseline in laps or angle
        rl_total_progress = self.rl_car.laps_completed + self.rl_car.angle / (2 * math.pi)
        baseline_total_progress = self.baseline_car.laps_completed + self.baseline_car.angle / (2 * math.pi)
        
        if rl_total_progress > baseline_total_progress:
            reward += 0.5
        else:
            reward -= 0.2
            
        # Reward for using acceleration tiles effectively
        if hasattr(self.rl_car, 'speed_multiplier') and self.rl_car.speed_multiplier > 1.0:
            reward += 0.3
            
        # Penalty for deceleration tiles
        if hasattr(self.rl_car, 'speed_multiplier') and self.rl_car.speed_multiplier < 1.0:
            reward -= 0.3
            
        # Terminal rewards
        if terminated:
            if winner == "RL":
                reward += 100.0  # Big reward for winning
            elif winner == "Baseline":
                reward -= 50.0   # Penalty for losing
            else:
                reward += 0.0    # Neutral for timeout
                
        return reward
        
    def _check_race_finished(self) -> Tuple[bool, str]:
        """Check if race is finished and determine winner based on laps"""
        
        # Check if either car completed required laps
        if self.rl_car.laps_completed >= self.track.laps_to_win:
            return True, "RL"
        elif self.baseline_car.laps_completed >= self.track.laps_to_win:
            return True, "Baseline"
                    
        # Check for timeout
        max_race_time = 180.0  # 3 minutes max for circular track
        if self.race_time >= max_race_time:
            # Determine winner by total progress
            rl_progress = self.rl_car.laps_completed + self.rl_car.angle / (2 * math.pi)
            baseline_progress = self.baseline_car.laps_completed + self.baseline_car.angle / (2 * math.pi)
            
            if rl_progress > baseline_progress:
                return True, "RL"
            else:
                return True, "Baseline"
                
        return False, None
        
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'race_time': self.race_time,
            'rl_car_laps': self.rl_car.laps_completed,
            'baseline_car_laps': self.baseline_car.laps_completed,
            'rl_car_angle': self.rl_car.angle,
            'baseline_car_angle': self.baseline_car.angle,
            'rl_car_lane': self.rl_car.current_lane,
            'rl_car_angular_speed': self.rl_car.angular_velocity,
            'episode': self.episode,
            'race_finished': self.race_finished,
            'winner': self.winner
        }
        
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and self.renderer:
            self.renderer.render(
                self.track, self.cars, self.tile_manager, 
                self.race_time, self.episode
            )
            
    def close(self):
        """Clean up environment resources"""
        if self.renderer:
            self.renderer.cleanup()
            
    def get_action_meanings(self) -> List[str]:
        """Get human-readable action names"""
        return ["Stay", "Inner Lane", "Outer Lane", "Accelerate", "Decelerate"]