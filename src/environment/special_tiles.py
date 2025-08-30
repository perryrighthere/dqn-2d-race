import numpy as np
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
    def __init__(self, x: float, y: float, tile_type: TileType, 
                 size: float = TILE_SIZE, duration: float = 60.0):
        self.x = x  # Center x position
        self.y = y  # Center y position
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
        
    def get_bounding_box(self) -> Tuple[float, float, float, float]:
        """Get tile bounding box for collision detection"""
        half_size = self.size / 2
        return (
            self.x - half_size,  # left
            self.y - half_size,  # top
            self.x + half_size,  # right
            self.y + half_size   # bottom
        )
        
    def is_colliding_with_car(self, car) -> bool:
        """Check if this tile is colliding with a car"""
        if not self.active:
            return False
            
        tile_left, tile_top, tile_right, tile_bottom = self.get_bounding_box()
        car_left, car_top, car_right, car_bottom = car.get_bounding_box()
        
        return (car_left < tile_right and car_right > tile_left and
                car_top < tile_bottom and car_bottom > tile_top)
                
    def apply_effect_to_car(self, car):
        """Apply tile effect to a car"""
        if self.active and self.is_colliding_with_car(car):
            car.apply_tile_effect(self.tile_type.value, self.duration)
            return True
        return False

class TileManager:
    """Manages all special tiles on the track"""
    
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
            
            # Generate tiles along track length
            position = 100  # Start after some initial distance
            while position < self.track.length - 100:  # End before finish
                # Random spacing between tiles
                position += np.random.exponential(1.0 / density) * 100
                
                if position >= self.track.length - 100:
                    break
                    
                # Random tile type
                tile_type = np.random.choice([TileType.ACCELERATION, TileType.DECELERATION])
                
                # Create tile at lane center
                tile = SpecialTile(
                    x=lane.center,
                    y=position,
                    tile_type=tile_type
                )
                
                self.tiles.append(tile)
                self.tiles_by_lane[lane.lane_id].append(tile)
                
    def get_tiles_in_range(self, y_start: float, y_end: float, 
                          lane_id: int = None) -> List[SpecialTile]:
        """Get all tiles within a y-position range, optionally filtered by lane"""
        result = []
        
        for tile in self.tiles:
            if y_start <= tile.y <= y_end:
                if lane_id is None or self.track.get_lane_by_position(tile.x).lane_id == lane_id:
                    result.append(tile)
                    
        return result
        
    def get_tiles_near_car(self, car, look_ahead: float = 150.0, 
                          look_behind: float = 50.0) -> List[SpecialTile]:
        """Get tiles near a car's current position"""
        return self.get_tiles_in_range(
            car.y - look_behind,
            car.y + look_ahead
        )
        
    def check_car_collisions(self, car) -> List[SpecialTile]:
        """Check for collisions between car and tiles, return colliding tiles"""
        colliding_tiles = []
        nearby_tiles = self.get_tiles_near_car(car, look_ahead=50.0, look_behind=10.0)
        
        for tile in nearby_tiles:
            if tile.apply_effect_to_car(car):
                colliding_tiles.append(tile)
                
        return colliding_tiles
        
    def get_tile_info_for_lane(self, lane_id: int, y_position: float, 
                              look_ahead: float = 200.0) -> Dict:
        """Get information about tiles in a specific lane ahead of position"""
        if lane_id not in self.tiles_by_lane:
            return {'has_tile': False, 'distance': float('inf'), 'type': None}
            
        lane_tiles = self.tiles_by_lane[lane_id]
        upcoming_tiles = [tile for tile in lane_tiles 
                         if y_position < tile.y <= y_position + look_ahead]
        
        if not upcoming_tiles:
            return {'has_tile': False, 'distance': float('inf'), 'type': None}
            
        # Find closest tile
        closest_tile = min(upcoming_tiles, key=lambda t: t.y - y_position)
        
        return {
            'has_tile': True,
            'distance': closest_tile.y - y_position,
            'type': closest_tile.tile_type.value,
            'effect_multiplier': closest_tile.get_effect_multiplier(),
            'tile_object': closest_tile
        }
        
    def get_state_representation(self, car, max_look_ahead: float = 200.0) -> List[float]:
        """Get tile state representation for RL agent"""
        state = []
        
        # For each lane, get info about the closest tile
        for lane_id in range(self.track.num_lanes):
            tile_info = self.get_tile_info_for_lane(lane_id, car.y, max_look_ahead)
            
            if tile_info['has_tile']:
                # Normalize distance
                normalized_distance = tile_info['distance'] / max_look_ahead
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