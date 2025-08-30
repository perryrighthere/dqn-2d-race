import numpy as np
import math
from typing import List, Tuple, Dict
from enum import Enum
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.game_config import *

class TileType(Enum):
    ACCELERATION = "acceleration"
    DECELERATION = "deceleration"

class SpecialTile:
    def __init__(self, angle: float, radius: float, tile_type: TileType, 
                 size: float = TILE_SIZE, duration: float = 60.0):
        self.angle = angle  # Angular position on track
        self.radius = radius  # Radial position (lane center)
        self.tile_type = tile_type
        self.size = size
        self.duration = duration  # How long the effect lasts
        self.active = True
        
    def get_effect_multiplier(self) -> float:
        """Get speed multiplier for this tile type"""
        if self.tile_type == TileType.ACCELERATION:
            return ACCELERATION_BOOST
        elif self.tile_type == TileType.DECELERATION:
            return DECELERATION_FACTOR
        return 1.0
        
    def get_cartesian_position(self) -> Tuple[float, float]:
        """Get tile position in cartesian coordinates"""
        x = self.radius * math.cos(self.angle)
        y = self.radius * math.sin(self.angle)
        return x, y
        
    def is_colliding_with_car(self, car) -> bool:
        """Check if this tile is colliding with a car using angular/radial distance"""
        if not self.active:
            return False
            
        return car.is_colliding_with_tile(self.angle, self.radius, self.size)
                
    def apply_effect_to_car(self, car):
        """Apply tile effect to a car"""
        if self.active and self.is_colliding_with_car(car):
            car.apply_tile_effect(self.tile_type.value, self.duration)
            return True
        return False

class TileManager:
    """Manages all special tiles on the circular track"""
    
    def __init__(self, track):
        self.track = track
        self.tiles = []  # List of SpecialTile objects
        self.tiles_by_lane = {}  # Dict mapping lane_id to list of tiles
        
    def generate_tiles(self, density: float = TILE_DENSITY, seed: int = 42):
        """Generate special tiles for all lanes except middle lane"""
        np.random.seed(seed)
        self.clear_tiles()
        
        for lane in self.track.lanes:
            # Skip middle lane - must stay clear for baseline agent
            if self.track.is_middle_lane(lane.lane_id):
                continue
                
            self.tiles_by_lane[lane.lane_id] = []
            
            # Generate tiles around the circular track
            num_tiles = int(density * 15)  # Adjust for circular track
            for _ in range(num_tiles):
                angle = np.random.uniform(0, 2 * math.pi)
                tile_type = np.random.choice([TileType.ACCELERATION, TileType.DECELERATION])
                
                # Create tile at lane center radius
                tile = SpecialTile(
                    angle=angle,
                    radius=lane.center_radius,
                    tile_type=tile_type
                )
                
                self.tiles.append(tile)
                self.tiles_by_lane[lane.lane_id].append(tile)
                
    def get_tiles_in_angular_range(self, start_angle: float, end_angle: float, 
                                 lane_id: int = None) -> List[SpecialTile]:
        """Get all tiles within an angular range, optionally filtered by lane"""
        result = []
        
        for tile in self.tiles:
            # Check if tile is in angular range (handle wrapping)
            in_range = False
            if start_angle <= end_angle:
                in_range = start_angle <= tile.angle <= end_angle
            else:  # Wraps around 0
                in_range = tile.angle >= start_angle or tile.angle <= end_angle
                
            if in_range:
                if lane_id is None or self.track.get_lane_by_radius(tile.radius) == lane_id:
                    result.append(tile)
                    
        return result
        
    def get_tiles_near_car(self, car, look_ahead_angle: float = 0.3) -> List[SpecialTile]:
        """Get tiles near a car's current angular position"""
        start_angle = car.angle
        end_angle = (car.angle + look_ahead_angle) % (2 * math.pi)
        return self.get_tiles_in_angular_range(start_angle, end_angle)
        
    def check_car_collisions(self, car) -> List[SpecialTile]:
        """Check for collisions between car and tiles, return colliding tiles"""
        colliding_tiles = []
        nearby_tiles = self.get_tiles_near_car(car, look_ahead_angle=0.1)
        
        for tile in nearby_tiles:
            if tile.apply_effect_to_car(car):
                colliding_tiles.append(tile)
                
        return colliding_tiles
        
    def get_tile_info_for_lane(self, lane_id: int, car_angle: float, 
                              look_ahead_angle: float = 0.5) -> Dict:
        """Get information about tiles in a specific lane ahead of position"""
        if lane_id not in self.tiles_by_lane:
            return {'has_tile': False, 'distance': float('inf'), 'type': None}
            
        lane_tiles = self.tiles_by_lane[lane_id]
        
        # Find tiles in the look ahead range
        upcoming_tiles = []
        for tile in lane_tiles:
            # Calculate angular distance (considering wrapping)
            angle_diff = tile.angle - car_angle
            if angle_diff < 0:
                angle_diff += 2 * math.pi
            if angle_diff <= look_ahead_angle:
                upcoming_tiles.append((tile, angle_diff))
        
        if not upcoming_tiles:
            return {'has_tile': False, 'distance': float('inf'), 'type': None}
            
        # Find closest tile
        closest_tile, closest_distance = min(upcoming_tiles, key=lambda x: x[1])
        
        return {
            'has_tile': True,
            'distance': closest_distance,
            'type': closest_tile.tile_type.value,
            'effect_multiplier': closest_tile.get_effect_multiplier(),
            'tile_object': closest_tile
        }
        
    def get_state_representation(self, car, max_look_ahead_angle: float = 0.5) -> List[float]:
        """Get tile state representation for RL agent"""
        state = []
        
        # For each lane, get info about the closest tile
        for lane_id in range(self.track.num_lanes):
            tile_info = self.get_tile_info_for_lane(lane_id, car.angle, max_look_ahead_angle)
            
            if tile_info['has_tile']:
                # Normalize angular distance
                normalized_distance = tile_info['distance'] / max_look_ahead_angle
                # Encode tile type: 1.0 for acceleration, -1.0 for deceleration
                tile_encoding = 1.0 if tile_info['type'] == 'acceleration' else -1.0
                state.extend([normalized_distance, tile_encoding])
            else:
                # No tile in range
                state.extend([1.0, 0.0])
                
        return state
        
    def clear_tiles(self):
        """Clear all tiles"""
        self.tiles.clear()
        self.tiles_by_lane.clear()
        
    def reset(self, seed: int = None):
        """Reset tile manager and regenerate tiles"""
        if seed is not None:
            self.generate_tiles(seed=seed)
        else:
            self.generate_tiles()
            
    def get_tile_count(self) -> Dict[str, int]:
        """Get count of tiles by type"""
        counts = {'acceleration': 0, 'deceleration': 0}
        for tile in self.tiles:
            counts[tile.tile_type.value] += 1
        return counts