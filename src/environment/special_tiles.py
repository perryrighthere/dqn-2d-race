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
        
    def generate_tiles(self, density: float = TILE_DENSITY, seed: int = 42, accel_ratio: float = 0.6):
        """Generate special tiles for all lanes except middle lane with enhanced randomization"""
        np.random.seed(seed)
        self.clear_tiles()
        
        for lane in self.track.lanes:
            # Skip middle lane - must stay clear for baseline agent
            if self.track.is_middle_lane(lane.lane_id):
                continue
                
            self.tiles_by_lane[lane.lane_id] = []
            
            # Generate more tiles with better distribution
            base_tiles = int(density * 25)  # Increased base number
            # Add some randomness to tile count per lane
            tile_variation = np.random.randint(-3, 8)  # Can add 0-7 extra tiles
            num_tiles = max(base_tiles + tile_variation, 5)  # Minimum 5 tiles per lane
            
            # Keep track of all placed tiles to avoid spatial overlap
            placed_tiles = []  # List of (angle, radius, size) tuples
            
            for i in range(num_tiles):
                attempts = 0
                while attempts < 50:  # Prevent infinite loops
                    angle = np.random.uniform(0, 2 * math.pi)
                    
                    # Add some radial variation within lane for more realism
                    radius_variation = np.random.uniform(-8, 8)  # Small radius offset
                    actual_radius = max(lane.inner_radius + 10, 
                                      min(lane.outer_radius - 10, 
                                          lane.center_radius + radius_variation))
                    
                    # Create candidate tile size
                    candidate_size = TILE_SIZE + np.random.uniform(-5, 5)  # Slight size variation
                    
                    # Check for spatial overlap with all existing tiles
                    overlapping = False
                    for existing_angle, existing_radius, existing_size in placed_tiles:
                        # Calculate cartesian distance between tile centers
                        # d = sqrt(r1² + r2² - 2*r1*r2*cos(θ2 - θ1))
                        angle_diff = angle - existing_angle
                        cartesian_distance = math.sqrt(
                            actual_radius**2 + existing_radius**2 - 
                            2 * actual_radius * existing_radius * math.cos(angle_diff)
                        )
                        
                        # Minimum separation should be sum of half-sizes plus small buffer
                        min_separation = (candidate_size + existing_size) / 2 + 5  # 5 unit buffer
                        
                        if cartesian_distance < min_separation:
                            overlapping = True
                            break
                    
                    if not overlapping:
                        placed_tiles.append((angle, actual_radius, candidate_size))
                        break
                    attempts += 1
                
                # If we couldn't find a non-overlapping spot, skip this tile
                if attempts >= 50:
                    continue  # Skip this tile instead of forcing placement
                
                # Get the successfully placed tile parameters
                angle, actual_radius, candidate_size = placed_tiles[-1]
                
                # Configurable ratio of acceleration vs deceleration tiles
                accel_ratio = float(np.clip(accel_ratio if accel_ratio is not None else 0.6, 0.0, 1.0))
                tile_type = np.random.choice(
                    [TileType.ACCELERATION, TileType.DECELERATION], p=[accel_ratio, 1.0 - accel_ratio]
                )
                
                # Create tile with variations
                tile = SpecialTile(
                    angle=angle,
                    radius=actual_radius,
                    tile_type=tile_type,
                    size=candidate_size  # Use the calculated size
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
        
    def reset(self, seed: int = None, density: float = None, accel_ratio: float = None):
        """Reset tile manager and regenerate tiles"""
        if seed is not None:
            self.generate_tiles(
                density=density if density is not None else TILE_DENSITY,
                seed=seed,
                accel_ratio=accel_ratio if accel_ratio is not None else 0.6,
            )
        else:
            self.generate_tiles(
                density=density if density is not None else TILE_DENSITY,
                accel_ratio=accel_ratio if accel_ratio is not None else 0.6,
            )
            
    def get_tile_count(self) -> Dict[str, int]:
        """Get count of tiles by type"""
        counts = {'acceleration': 0, 'deceleration': 0}
        for tile in self.tiles:
            counts[tile.tile_type.value] += 1
        return counts
    
    def get_detailed_tile_stats(self) -> Dict:
        """Get detailed statistics about tile placement"""
        stats = {
            'total_tiles': len(self.tiles),
            'tiles_by_type': self.get_tile_count(),
            'tiles_by_lane': {},
            'average_tiles_per_lane': 0
        }
        
        # Count tiles per lane
        for lane_id, tiles in self.tiles_by_lane.items():
            lane_stats = {
                'total': len(tiles),
                'acceleration': sum(1 for t in tiles if t.tile_type == TileType.ACCELERATION),
                'deceleration': sum(1 for t in tiles if t.tile_type == TileType.DECELERATION)
            }
            stats['tiles_by_lane'][f'lane_{lane_id}'] = lane_stats
        
        # Calculate average
        non_middle_lanes = len([lid for lid in self.tiles_by_lane.keys() 
                               if not self.track.is_middle_lane(lid)])
        if non_middle_lanes > 0:
            stats['average_tiles_per_lane'] = stats['total_tiles'] / non_middle_lanes
        
        return stats
