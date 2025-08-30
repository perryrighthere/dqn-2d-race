import numpy as np
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.game_config import *

class Lane:
    def __init__(self, lane_id: int, left_boundary: float, right_boundary: float):
        self.lane_id = lane_id
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary
        self.center = (left_boundary + right_boundary) / 2
        self.width = right_boundary - left_boundary
        self.special_tiles = []  # List of (position, tile_type) tuples
        
    def add_special_tile(self, position: float, tile_type: str):
        """Add a special tile to this lane"""
        self.special_tiles.append((position, tile_type))
        
    def get_tiles_in_range(self, start_pos: float, end_pos: float) -> List[Tuple[float, str]]:
        """Get all special tiles within a position range"""
        return [(pos, tile_type) for pos, tile_type in self.special_tiles 
                if start_pos <= pos <= end_pos]
    
    def is_position_in_lane(self, x_position: float) -> bool:
        """Check if x position is within this lane boundaries"""
        return self.left_boundary <= x_position <= self.right_boundary

class Track:
    def __init__(self, length: int = TRACK_LENGTH, width: int = TRACK_WIDTH, num_lanes: int = NUM_LANES):
        self.length = length
        self.width = width
        self.num_lanes = num_lanes
        self.lane_width = width // num_lanes
        
        # Create lanes
        self.lanes = []
        for i in range(num_lanes):
            left_boundary = i * self.lane_width
            right_boundary = (i + 1) * self.lane_width
            lane = Lane(i, left_boundary, right_boundary)
            self.lanes.append(lane)
            
        self.middle_lane_id = num_lanes // 2  # Middle lane (0-indexed)
        
    def get_lane_by_id(self, lane_id: int) -> Lane:
        """Get lane object by ID"""
        if 0 <= lane_id < self.num_lanes:
            return self.lanes[lane_id]
        raise ValueError(f"Invalid lane ID: {lane_id}")
        
    def get_lane_by_position(self, x_position: float) -> int:
        """Get lane ID based on x position"""
        for lane in self.lanes:
            if lane.is_position_in_lane(x_position):
                return lane.lane_id
        # If outside track boundaries, return closest lane
        if x_position < 0:
            return 0
        elif x_position >= self.width:
            return self.num_lanes - 1
        return -1
        
    def get_lane_center(self, lane_id: int) -> float:
        """Get center x-coordinate of a lane"""
        return self.get_lane_by_id(lane_id).center
        
    def is_middle_lane(self, lane_id: int) -> bool:
        """Check if lane is the middle lane"""
        return lane_id == self.middle_lane_id
        
    def get_valid_lane_changes(self, current_lane: int) -> List[int]:
        """Get list of valid lanes to move to from current lane"""
        valid_lanes = [current_lane]  # Can always stay in current lane
        
        # Can move left
        if current_lane > 0:
            valid_lanes.append(current_lane - 1)
            
        # Can move right
        if current_lane < self.num_lanes - 1:
            valid_lanes.append(current_lane + 1)
            
        return valid_lanes
        
    def generate_special_tiles(self, density: float = TILE_DENSITY):
        """Generate special tiles for all lanes except middle lane"""
        np.random.seed(42)  # For reproducible tile generation
        
        for lane in self.lanes:
            # Skip middle lane - must stay clear
            if self.is_middle_lane(lane.lane_id):
                continue
                
            # Generate tiles along track length
            position = 0
            while position < self.length:
                # Random spacing between tiles
                position += np.random.exponential(1.0 / density) * 100
                
                if position >= self.length:
                    break
                    
                # Random tile type
                tile_type = np.random.choice(['acceleration', 'deceleration'])
                lane.add_special_tile(position, tile_type)
                
    def get_tiles_near_position(self, y_position: float, look_ahead: float = 100) -> dict:
        """Get all tiles within look_ahead distance from current position"""
        tiles_by_lane = {}
        
        for lane in self.lanes:
            tiles = lane.get_tiles_in_range(y_position, y_position + look_ahead)
            if tiles:
                tiles_by_lane[lane.lane_id] = tiles
                
        return tiles_by_lane
        
    def reset(self):
        """Reset track state and regenerate tiles"""
        for lane in self.lanes:
            lane.special_tiles.clear()
        self.generate_special_tiles()