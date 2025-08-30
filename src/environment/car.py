import numpy as np
from typing import Tuple, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.game_config import *

class Car:
    def __init__(self, x: float, y: float, width: int = CAR_WIDTH, height: int = CAR_HEIGHT):
        # Position
        self.x = x  # Horizontal position (lane position)
        self.y = y  # Vertical position (progress along track)
        self.width = width
        self.height = height
        
        # Physics
        self.velocity_x = 0.0  # Horizontal velocity (for lane changing)
        self.velocity_y = BASE_SPEED  # Vertical velocity (forward motion)
        self.acceleration_x = 0.0
        self.acceleration_y = 0.0
        
        # Lane information
        self.current_lane = 0
        self.target_lane = 0
        self.lane_change_speed = 3.0  # Speed of lane transitions
        
        # Special tile effects
        self.speed_multiplier = 1.0
        self.tile_effect_duration = 0
        
        # Constraints
        self.max_speed = MAX_SPEED
        self.min_speed = MIN_SPEED
        
    def update_physics(self, dt: float = 1.0):
        """Update car physics"""
        # Apply acceleration
        self.velocity_x += self.acceleration_x * dt
        self.velocity_y += self.acceleration_y * dt
        
        # Apply speed constraints
        self.velocity_y = np.clip(self.velocity_y, self.min_speed, self.max_speed)
        
        # Apply special tile speed multiplier
        effective_speed_y = self.velocity_y * self.speed_multiplier
        
        # Update position
        self.x += self.velocity_x * dt
        self.y += effective_speed_y * dt
        
        # Handle lane changing physics
        self.handle_lane_change(dt)
        
        # Reduce tile effect duration
        if self.tile_effect_duration > 0:
            self.tile_effect_duration -= dt
            if self.tile_effect_duration <= 0:
                self.speed_multiplier = 1.0
                
    def handle_lane_change(self, dt: float):
        """Handle smooth lane transitions"""
        if self.current_lane != self.target_lane:
            # Calculate target x position for the lane
            from .track import Track
            track = Track()  # Temporary track instance for lane center calculation
            target_x = track.get_lane_center(self.target_lane)
            
            # Move towards target lane
            dx = target_x - self.x
            if abs(dx) > 1.0:  # Still moving towards target
                self.velocity_x = np.sign(dx) * self.lane_change_speed
            else:  # Close enough to target
                self.x = target_x
                self.velocity_x = 0.0
                self.current_lane = self.target_lane
                
    def change_lane(self, new_lane: int):
        """Initiate lane change"""
        self.target_lane = new_lane
        
    def apply_acceleration(self, acceleration: float):
        """Apply acceleration to forward motion"""
        self.acceleration_y = acceleration
        
    def apply_tile_effect(self, tile_type: str, duration: float = 60.0):
        """Apply special tile effect"""
        if tile_type == 'acceleration':
            self.speed_multiplier = ACCELERATION_BOOST
        elif tile_type == 'deceleration':
            self.speed_multiplier = DECELERATION_FACTOR
        
        self.tile_effect_duration = duration
        
    def get_position(self) -> Tuple[float, float]:
        """Get current position as tuple"""
        return (self.x, self.y)
        
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get bounding box for collision detection"""
        left = self.x - self.width // 2
        right = self.x + self.width // 2
        top = self.y - self.height // 2
        bottom = self.y + self.height // 2
        return (left, top, right, bottom)
        
    def is_colliding_with_tile(self, tile_x: float, tile_y: float, tile_size: float = TILE_SIZE) -> bool:
        """Check collision with a special tile"""
        car_left, car_top, car_right, car_bottom = self.get_bounding_box()
        
        tile_left = tile_x - tile_size // 2
        tile_right = tile_x + tile_size // 2
        tile_top = tile_y - tile_size // 2
        tile_bottom = tile_y + tile_size // 2
        
        return (car_left < tile_right and car_right > tile_left and
                car_top < tile_bottom and car_bottom > tile_top)
        
    def reset(self, x: float, y: float):
        """Reset car to initial state"""
        self.x = x
        self.y = y
        self.velocity_x = 0.0
        self.velocity_y = BASE_SPEED
        self.acceleration_x = 0.0
        self.acceleration_y = 0.0
        self.current_lane = 0
        self.target_lane = 0
        self.speed_multiplier = 1.0
        self.tile_effect_duration = 0

class BaselineCar(Car):
    """Baseline agent car - always maintains constant speed in middle lane"""
    
    def __init__(self, x: float, y: float):
        super().__init__(x, y)
        self.velocity_y = BASE_SPEED  # Fixed speed
        
    def update_physics(self, dt: float = 1.0):
        """Override to maintain constant behavior"""
        # Only update y position with constant speed
        self.y += self.velocity_y * dt
        
        # Stay in middle lane
        from .track import Track
        track = Track()
        middle_lane = track.num_lanes // 2
        self.x = track.get_lane_center(middle_lane)
        self.current_lane = middle_lane
        self.target_lane = middle_lane
        
    def apply_tile_effect(self, tile_type: str, duration: float = 60.0):
        """Baseline car is not affected by tiles in middle lane"""
        pass  # No effect for baseline car

class RLCar(Car):
    """RL agent car with full physics and tile interactions"""
    
    def __init__(self, x: float, y: float):
        super().__init__(x, y)
        
    def get_state(self, track) -> np.ndarray:
        """Get state representation for RL agent"""
        # Current position and velocity
        state = [
            self.x / track.width,  # Normalized x position
            self.y / track.length,  # Normalized y position
            self.velocity_y / self.max_speed,  # Normalized speed
            self.current_lane / (track.num_lanes - 1),  # Normalized lane
            self.speed_multiplier,  # Current speed multiplier
        ]
        
        # Add information about nearby tiles
        nearby_tiles = track.get_tiles_near_position(self.y, look_ahead=200)
        
        # For each lane, add info about closest tile
        for lane_id in range(track.num_lanes):
            if lane_id in nearby_tiles and nearby_tiles[lane_id]:
                closest_tile = min(nearby_tiles[lane_id], key=lambda x: x[0])
                tile_distance = (closest_tile[0] - self.y) / 200.0  # Normalized distance
                tile_type = 1.0 if closest_tile[1] == 'acceleration' else -1.0
                state.extend([tile_distance, tile_type])
            else:
                state.extend([1.0, 0.0])  # No tile in range
                
        return np.array(state, dtype=np.float32)