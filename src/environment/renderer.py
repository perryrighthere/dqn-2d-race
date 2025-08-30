import pygame
import numpy as np
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
        pygame.display.set_caption("2D Car Racing - DQN vs Baseline")
        
        self.clock = pygame.time.Clock()
        
        # Camera settings for following the race
        self.camera_y = 0
        self.track_view_width = 400  # Width of track view area
        self.track_view_height = height - 100  # Leave space for UI
        self.track_offset_x = 50  # Offset from left edge
        self.track_offset_y = 50   # Offset from top edge
        
        # Fonts
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
    def update_camera(self, cars: List[Car]):
        """Update camera position to follow the leading car"""
        if cars:
            # Follow the car that's furthest ahead
            leading_y = max(car.y for car in cars)
            self.camera_y = leading_y - self.track_view_height // 3
            
    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        # Scale track width to fit in track view area
        screen_x = int(self.track_offset_x + 
                      (world_x / TRACK_WIDTH) * self.track_view_width)
        
        # Y coordinate relative to camera
        screen_y = int(self.track_offset_y + self.track_view_height - 
                      (world_y - self.camera_y) * 0.5)  # Scale factor for y
        
        return screen_x, screen_y
        
    def render_track(self, track: Track):
        """Render the racing track with lanes"""
        # Draw track background
        track_rect = pygame.Rect(
            self.track_offset_x, self.track_offset_y,
            self.track_view_width, self.track_view_height
        )
        pygame.draw.rect(self.screen, GRAY, track_rect)
        
        # Draw lane dividers
        for i in range(1, track.num_lanes):
            lane_boundary = i * (self.track_view_width / track.num_lanes)
            x = self.track_offset_x + int(lane_boundary)
            
            # Draw dashed line
            for y in range(self.track_offset_y, 
                          self.track_offset_y + self.track_view_height, 20):
                if (y // 20) % 2 == 0:  # Dashed pattern
                    pygame.draw.line(self.screen, WHITE, (x, y), (x, y + 10), 2)
                    
        # Draw track borders
        pygame.draw.rect(self.screen, WHITE, track_rect, 3)
        
        # Draw start/finish line (if visible)
        start_screen_x, start_screen_y = self.world_to_screen(0, 0)
        finish_screen_x, finish_screen_y = self.world_to_screen(0, track.length)
        
        # Start line
        if 0 <= start_screen_y <= self.height:
            pygame.draw.line(self.screen, GREEN,
                           (self.track_offset_x, start_screen_y),
                           (self.track_offset_x + self.track_view_width, start_screen_y), 4)
            
        # Finish line
        if 0 <= finish_screen_y <= self.height:
            pygame.draw.line(self.screen, RED,
                           (self.track_offset_x, finish_screen_y),
                           (self.track_offset_x + self.track_view_width, finish_screen_y), 4)
                           
    def render_special_tiles(self, tile_manager: TileManager):
        """Render special tiles on the track"""
        for tile in tile_manager.tiles:
            screen_x, screen_y = self.world_to_screen(tile.x, tile.y)
            
            # Only render if tile is visible on screen
            if (screen_y > -tile.size and 
                screen_y < self.height + tile.size):
                
                # Choose color based on tile type
                if tile.tile_type == TileType.ACCELERATION:
                    color = GREEN
                else:  # DECELERATION
                    color = RED
                    
                # Draw tile as a diamond/square
                tile_size = int(tile.size * 0.3)  # Scale down for screen
                tile_rect = pygame.Rect(
                    screen_x - tile_size // 2,
                    screen_y - tile_size // 2,
                    tile_size,
                    tile_size
                )
                pygame.draw.rect(self.screen, color, tile_rect)
                pygame.draw.rect(self.screen, WHITE, tile_rect, 2)
                
                # Add symbol to indicate tile type
                if tile.tile_type == TileType.ACCELERATION:
                    # Draw "+" symbol
                    pygame.draw.line(self.screen, WHITE,
                                   (screen_x - 5, screen_y),
                                   (screen_x + 5, screen_y), 2)
                    pygame.draw.line(self.screen, WHITE,
                                   (screen_x, screen_y - 5),
                                   (screen_x, screen_y + 5), 2)
                else:
                    # Draw "-" symbol
                    pygame.draw.line(self.screen, WHITE,
                                   (screen_x - 5, screen_y),
                                   (screen_x + 5, screen_y), 2)
                    
    def render_car(self, car: Car, color: Tuple[int, int, int], label: str = ""):
        """Render a single car"""
        screen_x, screen_y = self.world_to_screen(car.x, car.y)
        
        # Only render if car is visible on screen
        if screen_y > -50 and screen_y < self.height + 50:
            # Draw car as rectangle
            car_width = int(car.width * 0.4)  # Scale down for screen
            car_height = int(car.height * 0.6)
            
            car_rect = pygame.Rect(
                screen_x - car_width // 2,
                screen_y - car_height // 2,
                car_width,
                car_height
            )
            
            pygame.draw.rect(self.screen, color, car_rect)
            pygame.draw.rect(self.screen, WHITE, car_rect, 2)
            
            # Draw speed multiplier effect
            if hasattr(car, 'speed_multiplier') and car.speed_multiplier != 1.0:
                effect_color = GREEN if car.speed_multiplier > 1.0 else RED
                pygame.draw.circle(self.screen, effect_color,
                                 (screen_x, screen_y - 20), 8)
                                 
            # Draw label
            if label:
                text = self.small_font.render(label, True, WHITE)
                self.screen.blit(text, (screen_x - 15, screen_y + 15))
                
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
        """Render UI information"""
        ui_x = self.track_offset_x + self.track_view_width + 20
        ui_y = self.track_offset_y
        
        # Race information
        race_info = [
            f"Episode: {episode}",
            f"Race Time: {race_time:.1f}s",
            f"Camera Y: {self.camera_y:.0f}",
            "",
            "Cars:"
        ]
        
        # Car information
        for i, car in enumerate(cars):
            car_type = "BASELINE" if isinstance(car, BaselineCar) else "RL AGENT"
            speed = car.velocity_y * car.speed_multiplier if hasattr(car, 'speed_multiplier') else car.velocity_y
            
            car_info = [
                f"{car_type}:",
                f"  Y: {car.y:.0f}",
                f"  Speed: {speed:.1f}",
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
                self.screen.blit(rendered_text, (ui_x, ui_y + i * 20))
                
    def render_progress_bar(self, cars: List[Car], track_length: float):
        """Render progress bars for each car"""
        bar_x = self.track_offset_x + self.track_view_width + 20
        bar_y = self.height - 150
        bar_width = 200
        bar_height = 20
        
        for i, car in enumerate(cars):
            # Background bar
            bar_rect = pygame.Rect(bar_x, bar_y + i * 30, bar_width, bar_height)
            pygame.draw.rect(self.screen, GRAY, bar_rect)
            pygame.draw.rect(self.screen, WHITE, bar_rect, 2)
            
            # Progress fill
            progress = min(car.y / track_length, 1.0)
            fill_width = int(bar_width * progress)
            
            if fill_width > 0:
                fill_rect = pygame.Rect(bar_x, bar_y + i * 30, fill_width, bar_height)
                color = BLUE if isinstance(car, BaselineCar) else RED
                pygame.draw.rect(self.screen, color, fill_rect)
                
            # Progress text
            progress_text = f"{progress * 100:.1f}%"
            text_surface = self.small_font.render(progress_text, True, WHITE)
            self.screen.blit(text_surface, (bar_x + bar_width + 10, bar_y + i * 30))
            
    def render(self, track: Track, cars: List[Car], tile_manager: TileManager,
               race_time: float, episode: int = 0):
        """Main render function"""
        # Clear screen
        self.screen.fill(BLACK)
        
        # Update camera to follow cars
        self.update_camera(cars)
        
        # Render components
        self.render_track(track)
        self.render_special_tiles(tile_manager)
        self.render_cars(cars)
        self.render_ui(cars, race_time, episode)
        self.render_progress_bar(cars, track.length)
        
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