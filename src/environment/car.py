import numpy as np
import math
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.game_config import *

class Car:
    def __init__(self, angle: float, lane: int, width: int = CAR_WIDTH, height: int = CAR_HEIGHT):
        # Position - using polar coordinates
        self.angle = angle  # Angular position around track (0 to 2π)
        self.radius = 0  # Will be set based on lane
        self.width = width
        self.height = height
        
        # Physics
        self.angular_velocity = BASE_SPEED / TRACK_RADIUS  # Angular velocity (radians per time)
        self.angular_acceleration = 0.0
        
        # Lane information
        self.current_lane = lane
        self.target_lane = lane
        self.lane_change_progress = 1.0  # 0.0 = changing, 1.0 = in target lane
        self.lane_change_speed = 0.1  # Speed of lane transitions (per frame)
        
        # Lap tracking
        self.laps_completed = 0
        self.total_angle = angle  # Total angle traversed (including multiple laps)
        
        # Special tile effects
        self.speed_multiplier = 1.0
        self.tile_effect_duration = 0
        
        # Constraints
        self.max_angular_speed = MAX_SPEED / TRACK_RADIUS
        self.min_angular_speed = MIN_SPEED / TRACK_RADIUS
        
    def update_radius_from_lane(self, track):
        """Update radius based on current lane and lane change progress"""
        if self.lane_change_progress >= 1.0:
            # Fully in target lane
            self.radius = track.get_lane_center_radius(self.current_lane)
        else:
            # Transitioning between lanes
            current_radius = track.get_lane_center_radius(self.current_lane)
            target_radius = track.get_lane_center_radius(self.target_lane)
            self.radius = current_radius + (target_radius - current_radius) * (1.0 - self.lane_change_progress)
    
    def update_physics(self, dt: float = 1.0, track=None):
        """Update car physics"""
        # Apply angular acceleration
        self.angular_velocity += self.angular_acceleration * dt
        
        # Apply speed constraints
        self.angular_velocity = np.clip(self.angular_velocity, self.min_angular_speed, self.max_angular_speed)
        
        # Apply special tile speed multiplier
        effective_angular_velocity = self.angular_velocity * self.speed_multiplier
        
        # Update angular position
        old_angle = self.angle
        self.angle += effective_angular_velocity * dt
        self.total_angle += effective_angular_velocity * dt
        
        # Handle lap completion
        if old_angle > self.angle:  # Wrapped around from 2π to 0
            self.laps_completed += 1
            
        # Normalize angle to [0, 2π]
        self.angle = self.angle % (2 * math.pi)
        
        # Handle lane changing
        self.handle_lane_change(dt)
        
        # Update radius based on current lane
        if track:
            self.update_radius_from_lane(track)
        
        # Reduce tile effect duration
        if self.tile_effect_duration > 0:
            self.tile_effect_duration -= dt
            if self.tile_effect_duration <= 0:
                self.speed_multiplier = 1.0
                
    def handle_lane_change(self, dt: float):
        """Handle smooth lane transitions"""
        if self.current_lane != self.target_lane:
            # Progress the lane change
            self.lane_change_progress += self.lane_change_speed * dt
            
            if self.lane_change_progress >= 1.0:
                # Lane change complete
                self.lane_change_progress = 1.0
                self.current_lane = self.target_lane
                
    def change_lane(self, new_lane: int):
        """Initiate lane change"""
        if new_lane != self.target_lane:
            self.target_lane = new_lane
            self.lane_change_progress = 0.0
        
    def apply_acceleration(self, acceleration: float):
        """Apply acceleration to angular motion"""
        self.angular_acceleration = acceleration / TRACK_RADIUS
        
    def apply_tile_effect(self, tile_type: str, duration: float = 60.0):
        """Apply special tile effect"""
        if tile_type == 'acceleration':
            self.speed_multiplier = ACCELERATION_BOOST
        elif tile_type == 'deceleration':
            self.speed_multiplier = DECELERATION_FACTOR
        
        self.tile_effect_duration = duration
        
    def get_position(self) -> Tuple[float, float]:
        """Get current position as cartesian coordinates"""
        x = self.radius * math.cos(self.angle)
        y = self.radius * math.sin(self.angle)
        return (x, y)
        
    def get_polar_position(self) -> Tuple[float, float]:
        """Get current position as polar coordinates"""
        return (self.angle, self.radius)
        
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box for collision detection in cartesian coordinates"""
        x, y = self.get_position()
        left = x - self.width // 2
        right = x + self.width // 2
        top = y - self.height // 2
        bottom = y + self.height // 2
        return (left, top, right, bottom)
        
    def is_colliding_with_tile(self, tile_angle: float, tile_radius: float, tile_size: float = TILE_SIZE) -> bool:
        """Check collision with a special tile using angular distance"""
        # Simple collision based on angular and radial distance
        angle_diff = abs(self.angle - tile_angle)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff
        
        radius_diff = abs(self.radius - tile_radius)
        
        # Convert tile size to angular and radial tolerances
        angular_tolerance = tile_size / (2 * self.radius) if self.radius > 0 else 0.1
        radial_tolerance = tile_size / 2
        
        return angle_diff < angular_tolerance and radius_diff < radial_tolerance
        
    def reset(self, angle: float, lane: int):
        """Reset car to initial state"""
        self.angle = angle
        self.current_lane = lane
        self.target_lane = lane
        self.lane_change_progress = 1.0
        self.angular_velocity = BASE_SPEED / TRACK_RADIUS
        self.angular_acceleration = 0.0
        self.laps_completed = 0
        self.total_angle = angle
        self.speed_multiplier = 1.0
        self.tile_effect_duration = 0

class BaselineCar(Car):
    """Baseline agent car - always maintains constant speed in middle lane"""
    
    def __init__(self, angle: float, track):
        middle_lane = track.num_lanes // 2
        super().__init__(angle, middle_lane)
        self.angular_velocity = BASE_SPEED / TRACK_RADIUS  # Fixed angular speed
        
    def update_physics(self, dt: float = 1.0, track=None):
        """Override to maintain constant behavior"""
        # Only update angular position with constant speed
        old_angle = self.angle
        self.angle += self.angular_velocity * dt
        self.total_angle += self.angular_velocity * dt
        
        # Handle lap completion
        if old_angle > self.angle:  # Wrapped around
            self.laps_completed += 1
            
        # Normalize angle
        self.angle = self.angle % (2 * math.pi)
        
        # Always stay in middle lane
        if track:
            middle_lane = track.num_lanes // 2
            self.current_lane = middle_lane
            self.target_lane = middle_lane
            self.lane_change_progress = 1.0
            self.radius = track.get_lane_center_radius(middle_lane)
        
    def apply_tile_effect(self, tile_type: str, duration: float = 60.0):
        """Baseline car is not affected by tiles in middle lane"""
        pass  # No effect for baseline car

class RLCar(Car):
    """RL agent car with full physics and tile interactions"""
    
    def __init__(self, angle: float, lane: int):
        super().__init__(angle, lane)
        
    def get_state(self, track) -> np.ndarray:
        """Get state representation for RL agent"""
        # Current position and velocity
        state = [
            self.angle / (2 * math.pi),  # Normalized angular position
            self.radius / track.outer_radius,  # Normalized radial position
            self.angular_velocity / self.max_angular_speed,  # Normalized angular speed
            self.current_lane / (track.num_lanes - 1),  # Normalized lane
            self.speed_multiplier,  # Current speed multiplier
        ]
        
        # Add information about nearby tiles
        nearby_tiles = track.get_tiles_near_angle(self.angle, look_ahead_angle=0.5)
        
        # For each lane, add info about closest tile
        for lane_id in range(track.num_lanes):
            if lane_id in nearby_tiles and nearby_tiles[lane_id]:
                closest_tile = min(nearby_tiles[lane_id], key=lambda x: abs(x[0] - self.angle))
                # Calculate angular distance (handle wrapping)
                angle_diff = abs(closest_tile[0] - self.angle)
                if angle_diff > math.pi:
                    angle_diff = 2 * math.pi - angle_diff
                tile_distance = angle_diff / 0.5  # Normalized distance
                tile_type = 1.0 if closest_tile[1] == 'acceleration' else -1.0
                state.extend([tile_distance, tile_type])
            else:
                state.extend([1.0, 0.0])  # No tile in range
                
        return np.array(state, dtype=np.float32)