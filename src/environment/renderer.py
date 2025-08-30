import pygame
import numpy as np
import math
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from config.game_config import *
from .track import Track
from .car import Car, BaselineCar, RLCar
from .special_tiles import SpecialTile, TileType, TileManager

class GameRenderer:
    def __init__(self, width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("2D Car Racing - Circular Track DQN vs Baseline")
        
        self.clock = pygame.time.Clock()
        
        # Track display settings
        self.track_center_x = 300  # Center X of circular track on screen
        self.track_center_y = 300  # Center Y of circular track on screen
        self.display_scale = 0.7   # Scale factor for track display
        
        # UI area
        self.ui_x = 620  # X position for UI panel
        self.ui_y = 50   # Y position for UI panel
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def polar_to_screen(self, angle: float, radius: float) -> Tuple[int, int]:
        """Convert polar coordinates to screen coordinates"""
        # Apply display scale
        display_radius = radius * self.display_scale
        
        # Convert to cartesian coordinates
        x = display_radius * math.cos(angle)
        y = display_radius * math.sin(angle)
        
        # Translate to screen center
        screen_x = int(self.track_center_x + x)
        screen_y = int(self.track_center_y - y)  # Flip Y for screen coordinates
        
        return screen_x, screen_y
        
    def render_track(self, track: Track):
        """Render the circular racing track with concentric lanes"""
        # Draw track surface (fill between inner and outer boundaries)
        outer_radius = int(track.outer_radius * self.display_scale)
        inner_radius = int(track.inner_radius * self.display_scale)
        
        # Draw outer circle (track boundary)
        pygame.draw.circle(self.screen, GRAY, 
                         (self.track_center_x, self.track_center_y), 
                         outer_radius, 0)
        
        # Draw inner circle (track hole)
        pygame.draw.circle(self.screen, BLACK, 
                         (self.track_center_x, self.track_center_y), 
                         inner_radius, 0)
        
        # Draw lane dividers
        for i in range(1, track.num_lanes):
            lane_radius = int((track.inner_radius + i * track.lane_width) * self.display_scale)
            # Draw dashed circle for lane divider
            self.draw_dashed_circle(self.track_center_x, self.track_center_y, 
                                  lane_radius, WHITE, 2, 10, 10)
        
        # Draw track borders
        pygame.draw.circle(self.screen, WHITE, 
                         (self.track_center_x, self.track_center_y), 
                         outer_radius, 3)
        pygame.draw.circle(self.screen, WHITE, 
                         (self.track_center_x, self.track_center_y), 
                         inner_radius, 3)
        
        # Draw start/finish line at angle 0 (rightmost point)
        start_inner = self.polar_to_screen(0, track.inner_radius)
        start_outer = self.polar_to_screen(0, track.outer_radius)
        pygame.draw.line(self.screen, GREEN, start_inner, start_outer, 4)
        
    def draw_dashed_circle(self, center_x: int, center_y: int, radius: int, 
                          color: Tuple[int, int, int], width: int, 
                          dash_length: int, gap_length: int):
        """Draw a dashed circle"""
        circumference = 2 * math.pi * radius
        total_dash_cycle = dash_length + gap_length
        num_dashes = int(circumference / total_dash_cycle)
        
        for i in range(num_dashes):
            start_angle = (i * total_dash_cycle / radius)
            end_angle = ((i * total_dash_cycle + dash_length) / radius)
            
            # Draw arc for this dash
            start_x = int(center_x + radius * math.cos(start_angle))
            start_y = int(center_y - radius * math.sin(start_angle))
            end_x = int(center_x + radius * math.cos(end_angle))
            end_y = int(center_y - radius * math.sin(end_angle))
            
            # Approximate arc with line segments
            segments = max(3, int((end_angle - start_angle) * radius / 5))
            for j in range(segments):
                angle1 = start_angle + j * (end_angle - start_angle) / segments
                angle2 = start_angle + (j + 1) * (end_angle - start_angle) / segments
                
                x1 = int(center_x + radius * math.cos(angle1))
                y1 = int(center_y - radius * math.sin(angle1))
                x2 = int(center_x + radius * math.cos(angle2))
                y2 = int(center_y - radius * math.sin(angle2))
                
                pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width)
                           
    def render_special_tiles(self, tile_manager: TileManager):
        """Render special tiles on the circular track"""
        for tile in tile_manager.tiles:
            screen_x, screen_y = self.polar_to_screen(tile.angle, tile.radius)
            
            # Choose color based on tile type
            if tile.tile_type == TileType.ACCELERATION:
                color = GREEN
            else:  # DECELERATION
                color = RED
                
            # Draw tile as a diamond/square
            tile_size = int(tile.size * self.display_scale * 0.5)  # Scale for display
            tile_rect = pygame.Rect(
                screen_x - tile_size // 2,
                screen_y - tile_size // 2,
                tile_size,
                tile_size
            )
            pygame.draw.rect(self.screen, color, tile_rect)
            pygame.draw.rect(self.screen, WHITE, tile_rect, 1)
            
            # Add symbol to indicate tile type
            symbol_size = max(3, tile_size // 3)
            if tile.tile_type == TileType.ACCELERATION:
                # Draw "+" symbol
                pygame.draw.line(self.screen, WHITE,
                               (screen_x - symbol_size, screen_y),
                               (screen_x + symbol_size, screen_y), 2)
                pygame.draw.line(self.screen, WHITE,
                               (screen_x, screen_y - symbol_size),
                               (screen_x, screen_y + symbol_size), 2)
            else:
                # Draw "-" symbol
                pygame.draw.line(self.screen, WHITE,
                               (screen_x - symbol_size, screen_y),
                               (screen_x + symbol_size, screen_y), 2)
                    
    def render_car(self, car: Car, color: Tuple[int, int, int], label: str = ""):
        """Render a single car on the circular track"""
        screen_x, screen_y = self.polar_to_screen(car.angle, car.radius)
        
        # Draw car as circle (easier for circular track)
        car_radius = int(max(8, car.width * self.display_scale * 0.3))
        
        pygame.draw.circle(self.screen, color, (screen_x, screen_y), car_radius)
        pygame.draw.circle(self.screen, WHITE, (screen_x, screen_y), car_radius, 2)
        
        # Draw directional indicator (small line showing car's orientation)
        direction_length = car_radius + 5
        direction_angle = car.angle + math.pi / 2  # Perpendicular to radius
        dir_x = int(screen_x + direction_length * math.cos(direction_angle))
        dir_y = int(screen_y - direction_length * math.sin(direction_angle))
        pygame.draw.line(self.screen, WHITE, (screen_x, screen_y), (dir_x, dir_y), 2)
        
        # Draw speed multiplier effect
        if hasattr(car, 'speed_multiplier') and car.speed_multiplier != 1.0:
            effect_color = GREEN if car.speed_multiplier > 1.0 else RED
            pygame.draw.circle(self.screen, effect_color,
                             (screen_x, screen_y), car_radius + 6, 3)
                             
        # Draw label
        if label:
            text = self.small_font.render(label, True, WHITE)
            text_rect = text.get_rect(center=(screen_x, screen_y - car_radius - 15))
            self.screen.blit(text, text_rect)
                
    def render_cars(self, cars: List[Car]):
        """Render all cars"""
        for i, car in enumerate(cars):
            if isinstance(car, BaselineCar):
                self.render_car(car, BLUE, "BASE")
            elif isinstance(car, RLCar):
                self.render_car(car, RED, "RL")
            else:
                self.render_car(car, YELLOW, f"CAR{i}")
                
    def render_ui(self, cars: List[Car], race_time: float, episode: int = 0):
        """Render UI information for circular track"""
        # Race information
        race_info = [
            f"Episode: {episode}",
            f"Race Time: {race_time:.1f}s",
            f"Laps to Win: {LAPS_TO_WIN}",
            "",
            "Cars:"
        ]
        
        # Car information
        for i, car in enumerate(cars):
            car_type = "BASELINE" if isinstance(car, BaselineCar) else "RL AGENT"
            angular_speed = car.angular_velocity * car.speed_multiplier if hasattr(car, 'speed_multiplier') else car.angular_velocity
            linear_speed = angular_speed * car.radius if hasattr(car, 'radius') else 0
            
            car_info = [
                f"{car_type}:",
                f"  Laps: {car.laps_completed}",
                f"  Angle: {math.degrees(car.angle):.0f}°",
                f"  Speed: {linear_speed:.1f}",
                f"  Lane: {car.current_lane}",
                ""
            ]
            race_info.extend(car_info)
            
        # Render text
        for i, text in enumerate(race_info):
            if text:  # Skip empty strings
                color = WHITE
                if "BASELINE" in text:
                    color = BLUE
                elif "RL AGENT" in text:
                    color = RED
                    
                rendered_text = self.small_font.render(text, True, color)
                self.screen.blit(rendered_text, (self.ui_x, self.ui_y + i * 20))
                
    def render_progress_bar(self, cars: List[Car], track):
        """Render lap progress bars for each car"""
        bar_x = self.ui_x
        bar_y = self.height - 150
        bar_width = 160
        bar_height = 20
        
        for i, car in enumerate(cars):
            # Background bar
            bar_rect = pygame.Rect(bar_x, bar_y + i * 35, bar_width, bar_height)
            pygame.draw.rect(self.screen, GRAY, bar_rect)
            pygame.draw.rect(self.screen, WHITE, bar_rect, 2)
            
            # Progress calculation: laps completed + current lap progress
            lap_progress = car.angle / (2 * math.pi)
            total_progress = (car.laps_completed + lap_progress) / track.laps_to_win
            total_progress = min(total_progress, 1.0)
            
            fill_width = int(bar_width * total_progress)
            
            if fill_width > 0:
                fill_rect = pygame.Rect(bar_x, bar_y + i * 35, fill_width, bar_height)
                color = BLUE if isinstance(car, BaselineCar) else RED
                pygame.draw.rect(self.screen, color, fill_rect)
                
            # Progress text
            progress_text = f"{car.laps_completed}/{track.laps_to_win}"
            text_surface = self.small_font.render(progress_text, True, WHITE)
            self.screen.blit(text_surface, (bar_x, bar_y + i * 35 + bar_height + 5))
            
    def render(self, track: Track, cars: List[Car], tile_manager: TileManager,
               race_time: float, episode: int = 0):
        """Main render function for circular track"""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Render components
        self.render_track(track)
        self.render_special_tiles(tile_manager)
        self.render_cars(cars)
        self.render_ui(cars, race_time, episode)
        self.render_progress_bar(cars, track)
        
        # Update display
        pygame.display.flip()
        self.clock.tick(FPS)
        
    def handle_events(self) -> bool:
        """Handle pygame events, return False if quit requested"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
        return True
        
    def cleanup(self):
        """Clean up pygame resources"""
        pygame.quit()