import numpy as np
from typing import List, Tuple
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.game_config import *

class Lane:
    def __init__(self, lane_id: int, inner_radius: float, outer_radius: float):
        self.lane_id = lane_id
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        self.center_radius = (inner_radius + outer_radius) / 2
        self.width = outer_radius - inner_radius
        self.special_tiles = []  # List of (angle, tile_type) tuples for circular track
        
    def add_special_tile(self, angle: float, tile_type: str):
        """Add a special tile to this lane at given angle"""
        self.special_tiles.append((angle, tile_type))
        
    def get_tiles_in_angular_range(self, start_angle: float, end_angle: float) -> List[Tuple[float, str]]:
        """Get all special tiles within an angular range"""
        tiles = []
        for angle, tile_type in self.special_tiles:
            # Handle wrapping around 2π
            if start_angle <= end_angle:
                if start_angle <= angle <= end_angle:
                    tiles.append((angle, tile_type))
            else:  # Wraps around 0
                if angle >= start_angle or angle <= end_angle:
                    tiles.append((angle, tile_type))
        return tiles
    
    def is_radius_in_lane(self, radius: float) -> bool:
        """Check if radius is within this lane boundaries"""
        return self.inner_radius <= radius <= self.outer_radius
        
    def get_cartesian_position(self, angle: float) -> Tuple[float, float]:
        """Convert angle to cartesian coordinates at lane center"""
        x = self.center_radius * math.cos(angle)
        y = self.center_radius * math.sin(angle)
        return x, y

class Track:
    def __init__(self, radius: float = TRACK_RADIUS, track_width: float = TRACK_WIDTH, num_lanes: int = NUM_LANES):
        self.radius = radius  # Radius to center of track
        self.track_width = track_width
        self.num_lanes = num_lanes
        self.lane_width = track_width / num_lanes
        
        # Calculate track boundaries
        self.inner_radius = radius - track_width / 2
        self.outer_radius = radius + track_width / 2
        
        # Create circular lanes
        self.lanes = []
        for i in range(num_lanes):
            inner_radius = self.inner_radius + i * self.lane_width
            outer_radius = self.inner_radius + (i + 1) * self.lane_width
            lane = Lane(i, inner_radius, outer_radius)
            self.lanes.append(lane)
            
        self.middle_lane_id = num_lanes // 2  # Middle lane (0-indexed)
        
        # Track properties
        self.circumference = 2 * math.pi * self.radius
        self.laps_to_win = LAPS_TO_WIN
        
    def get_lane_by_id(self, lane_id: int) -> Lane:
        """Get lane object by ID"""
        if 0 <= lane_id < self.num_lanes:
            return self.lanes[lane_id]
        raise ValueError(f"Invalid lane ID: {lane_id}")
        
    def get_lane_by_radius(self, radius: float) -> int:
        """Get lane ID based on radius from center"""
        for lane in self.lanes:
            if lane.is_radius_in_lane(radius):
                return lane.lane_id
        # If outside track boundaries, return closest lane
        if radius < self.inner_radius:
            return 0
        elif radius > self.outer_radius:
            return self.num_lanes - 1
        return -1
        
    def get_lane_center_radius(self, lane_id: int) -> float:
        """Get center radius of a lane"""
        return self.get_lane_by_id(lane_id).center_radius
        
    def is_middle_lane(self, lane_id: int) -> bool:
        """Check if lane is the middle lane"""
        return lane_id == self.middle_lane_id
        
    def get_valid_lane_changes(self, current_lane: int) -> List[int]:
        """Get list of valid lanes to move to from current lane"""
        valid_lanes = [current_lane]  # Can always stay in current lane
        
        # Can move to inner lane (lower lane ID)
        if current_lane > 0:
            valid_lanes.append(current_lane - 1)
            
        # Can move to outer lane (higher lane ID) 
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
                
            # Generate tiles around the circle
            num_tiles = int(density * 20)  # Adjust tile count for circular track
            for _ in range(num_tiles):
                angle = np.random.uniform(0, 2 * math.pi)
                tile_type = np.random.choice(['acceleration', 'deceleration'])
                lane.add_special_tile(angle, tile_type)
                
    def get_tiles_near_angle(self, angle: float, look_ahead_angle: float = 0.5) -> dict:
        """Get all tiles within look_ahead_angle from current angle"""
        tiles_by_lane = {}
        
        # Calculate angular range with wrapping
        start_angle = angle
        end_angle = (angle + look_ahead_angle) % (2 * math.pi)
        
        for lane in self.lanes:
            tiles = lane.get_tiles_in_angular_range(start_angle, end_angle)
            if tiles:
                tiles_by_lane[lane.lane_id] = tiles
                
        return tiles_by_lane
        
    def normalize_angle(self, angle: float) -> float:
        """Normalize angle to [0, 2π] range"""
        while angle < 0:
            angle += 2 * math.pi
        while angle >= 2 * math.pi:
            angle -= 2 * math.pi
        return angle
        
    def angular_distance(self, angle1: float, angle2: float) -> float:
        """Calculate shortest angular distance between two angles"""
        diff = abs(angle2 - angle1)
        if diff > math.pi:
            diff = 2 * math.pi - diff
        return diff
        
    def angle_to_cartesian(self, angle: float, radius: float) -> Tuple[float, float]:
        """Convert polar coordinates to cartesian"""
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        return x, y
        
    def cartesian_to_angle(self, x: float, y: float) -> float:
        """Convert cartesian coordinates to angle"""
        return math.atan2(y, x)
        
    def reset(self, density: float = None):
        """Reset track state and regenerate tiles"""
        for lane in self.lanes:
            lane.special_tiles.clear()
        self.generate_special_tiles(density=density if density is not None else TILE_DENSITY)
