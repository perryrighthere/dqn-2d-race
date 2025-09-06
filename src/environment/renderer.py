import pygame
import numpy as np
import math
from typing import List, Tuple, Optional
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
        
        # Load and create background surfaces
        self._create_background_surfaces()
        
        # Car surface cache
        self._car_surfaces = {}
        self._create_car_surfaces()
    
    def _create_background_surfaces(self):
        """Create procedural background textures"""
        # Grass background
        self.grass_surface = pygame.Surface((self.width, self.height))
        self._draw_grass_pattern(self.grass_surface)
        
        # Track surface with asphalt texture
        self.track_surface = None
        
    def _draw_grass_pattern(self, surface):
        """Draw procedural grass pattern"""
        surface.fill((34, 139, 34))  # Forest green base
        
        # Add grass texture with random dots
        import random
        random.seed(42)  # Consistent grass pattern
        for _ in range(2000):
            x = random.randint(0, surface.get_width())
            y = random.randint(0, surface.get_height())
            # Varying shades of green
            grass_colors = [(50, 150, 50), (40, 120, 40), (60, 180, 60)]
            color = random.choice(grass_colors)
            pygame.draw.circle(surface, color, (x, y), random.randint(1, 3))
    
    def _create_track_surface(self, track: Track):
        """Create track surface with asphalt texture"""
        if self.track_surface is not None:
            return
            
        # Create surface large enough for the track
        size = int(track.outer_radius * 2 * self.display_scale) + 50
        self.track_surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        
        # Draw asphalt texture
        outer_radius = int(track.outer_radius * self.display_scale)
        inner_radius = int(track.inner_radius * self.display_scale)
        
        # Dark asphalt base
        pygame.draw.circle(self.track_surface, (45, 45, 45), (center, center), outer_radius)
        pygame.draw.circle(self.track_surface, (0, 0, 0, 0), (center, center), inner_radius)
        
        # Add asphalt texture with random noise
        import random
        random.seed(123)
        for _ in range(1000):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(inner_radius, outer_radius)
            x = int(center + radius * math.cos(angle))
            y = int(center + radius * math.sin(angle))
            if inner_radius < math.sqrt((x - center)**2 + (y - center)**2) < outer_radius:
                gray_val = random.randint(30, 60)
                color = (gray_val, gray_val, gray_val)
                pygame.draw.circle(self.track_surface, color, (x, y), random.randint(1, 2))
    
    def _create_car_surfaces(self):
        """Create enhanced car surfaces"""
        # DQN car (red sports car)
        self._car_surfaces['dqn'] = self._create_sports_car_surface((220, 20, 20), (255, 100, 100))
        
        # Baseline car (blue classic car)
        self._car_surfaces['baseline'] = self._create_classic_car_surface((20, 20, 220), (100, 100, 255))
    
    def _create_sports_car_surface(self, primary_color, accent_color):
        """Create a sports car surface"""
        size = 32
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        
        # Car body (elongated oval)
        pygame.draw.ellipse(surface, primary_color, (6, 10, 20, 12))
        pygame.draw.ellipse(surface, accent_color, (8, 11, 16, 10))
        
        # Car front (pointed)
        pygame.draw.polygon(surface, primary_color, [(26, 14), (30, 16), (26, 18)])
        
        # Windshield
        pygame.draw.ellipse(surface, (50, 50, 100), (10, 12, 10, 8))
        
        # Racing stripes
        pygame.draw.line(surface, WHITE, (8, 15), (24, 15), 1)
        
        return surface
    
    def _create_classic_car_surface(self, primary_color, accent_color):
        """Create a classic car surface"""
        size = 32
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        center = size // 2
        
        # Car body (rectangular with rounded corners)
        pygame.draw.rect(surface, primary_color, (8, 11, 18, 10), border_radius=3)
        pygame.draw.rect(surface, accent_color, (10, 12, 14, 8), border_radius=2)
        
        # Car front
        pygame.draw.rect(surface, primary_color, (26, 13, 4, 6), border_radius=2)
        
        # Windshield
        pygame.draw.rect(surface, (50, 50, 100), (12, 13, 8, 6))
        
        return surface
        
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
        """Render the circular racing track with enhanced graphics"""
        # Create track surface if needed
        self._create_track_surface(track)
        
        # Calculate track dimensions
        outer_radius = int(track.outer_radius * self.display_scale)
        inner_radius = int(track.inner_radius * self.display_scale)
        
        # Blit the textured track surface
        if self.track_surface:
            track_rect = self.track_surface.get_rect()
            track_rect.center = (self.track_center_x, self.track_center_y)
            self.screen.blit(self.track_surface, track_rect)
        
        # Draw lane dividers with improved styling
        for i in range(1, track.num_lanes):
            lane_radius = int((track.inner_radius + i * track.lane_width) * self.display_scale)
            # Draw dashed circle for lane divider (brighter white)
            self.draw_dashed_circle(self.track_center_x, self.track_center_y, 
                                  lane_radius, (255, 255, 255), 2, 8, 12)
        
        # Draw enhanced track borders with shadow effect
        # Outer border
        pygame.draw.circle(self.screen, (200, 200, 200), 
                         (self.track_center_x + 2, self.track_center_y + 2), 
                         outer_radius, 4)  # Shadow
        pygame.draw.circle(self.screen, WHITE, 
                         (self.track_center_x, self.track_center_y), 
                         outer_radius, 3)
        
        # Inner border
        pygame.draw.circle(self.screen, (200, 200, 200), 
                         (self.track_center_x + 2, self.track_center_y + 2), 
                         inner_radius, 4)  # Shadow
        pygame.draw.circle(self.screen, WHITE, 
                         (self.track_center_x, self.track_center_y), 
                         inner_radius, 3)
        
        # Draw enhanced start/finish line
        start_inner = self.polar_to_screen(0, track.inner_radius)
        start_outer = self.polar_to_screen(0, track.outer_radius)
        
        # Checkered pattern for start/finish line
        line_segments = 6
        for i in range(line_segments):
            progress = i / line_segments
            next_progress = (i + 1) / line_segments
            
            start_point = (
                int(start_inner[0] + (start_outer[0] - start_inner[0]) * progress),
                int(start_inner[1] + (start_outer[1] - start_inner[1]) * progress)
            )
            end_point = (
                int(start_inner[0] + (start_outer[0] - start_inner[0]) * next_progress),
                int(start_inner[1] + (start_outer[1] - start_inner[1]) * next_progress)
            )
            
            color = WHITE if i % 2 == 0 else BLACK
            pygame.draw.line(self.screen, color, start_point, end_point, 6)
        
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
        """Render enhanced special tiles on the circular track"""
        for tile in tile_manager.tiles:
            screen_x, screen_y = self.polar_to_screen(tile.angle, tile.radius)
            
            # Choose colors and effects based on tile type
            if tile.tile_type == TileType.ACCELERATION:
                primary_color = (0, 200, 0)  # Bright green
                secondary_color = (0, 255, 0)  # Neon green
                glow_color = (0, 100, 0)
                symbol = "+"
            else:  # DECELERATION
                primary_color = (200, 0, 0)  # Bright red
                secondary_color = (255, 0, 0)  # Neon red
                glow_color = (100, 0, 0)
                symbol = "-"
            
            # Draw tile with glow effect
            tile_size = int(tile.size * self.display_scale * 0.6)
            
            # Glow effect (multiple circles with decreasing alpha)
            for i in range(3):
                glow_radius = tile_size // 2 + i * 3
                glow_alpha = 60 - i * 20
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, (*glow_color, glow_alpha), 
                                 (glow_radius, glow_radius), glow_radius)
                self.screen.blit(glow_surface, 
                               (screen_x - glow_radius, screen_y - glow_radius))
            
            # Main tile (hexagonal shape)
            points = []
            for i in range(6):
                angle = i * math.pi / 3
                px = screen_x + (tile_size // 2) * math.cos(angle)
                py = screen_y + (tile_size // 2) * math.sin(angle)
                points.append((px, py))
            
            pygame.draw.polygon(self.screen, primary_color, points)
            pygame.draw.polygon(self.screen, secondary_color, points, 2)
            
            # Enhanced symbol with outline
            symbol_size = max(4, tile_size // 3)
            if symbol == "+":
                # Draw "+" with outline
                for offset in [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    color = BLACK if offset != (0, 0) else WHITE
                    x_off, y_off = offset
                    pygame.draw.line(self.screen, color,
                                   (screen_x - symbol_size + x_off, screen_y + y_off),
                                   (screen_x + symbol_size + x_off, screen_y + y_off), 3)
                    pygame.draw.line(self.screen, color,
                                   (screen_x + x_off, screen_y - symbol_size + y_off),
                                   (screen_x + x_off, screen_y + symbol_size + y_off), 3)
            else:
                # Draw "-" with outline
                for offset in [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                    color = BLACK if offset != (0, 0) else WHITE
                    x_off, y_off = offset
                    pygame.draw.line(self.screen, color,
                                   (screen_x - symbol_size + x_off, screen_y + y_off),
                                   (screen_x + symbol_size + x_off, screen_y + y_off), 3)
                    
    def render_car(self, car: Car, car_type: str = "default", label: str = ""):
        """Render an enhanced car on the circular track"""
        screen_x, screen_y = self.polar_to_screen(car.angle, car.radius)
        
        # Get appropriate car surface
        car_surface = self._car_surfaces.get(car_type)
        if car_surface is None:
            # Fallback to simple circle if surface not found
            car_radius = int(max(8, car.width * self.display_scale * 0.3))
            color = RED if car_type == 'dqn' else BLUE
            pygame.draw.circle(self.screen, color, (screen_x, screen_y), car_radius)
            pygame.draw.circle(self.screen, WHITE, (screen_x, screen_y), car_radius, 2)
        else:
            # Calculate car rotation based on track position
            # Car should face tangentially to the circular track for counter-clockwise motion
            # Car sprites are initially facing right (0°), so we need to rotate them to face the tangent
            tangent_angle = car.angle + math.pi / 2  # Tangent direction for counter-clockwise motion
            
            # Convert to degrees and adjust for pygame rotation
            # Pygame rotates counter-clockwise for positive angles, but we need to account for flipped Y
            rotation_degrees = math.degrees(tangent_angle)
            
            # Rotate car surface
            rotated_surface = pygame.transform.rotate(car_surface, -rotation_degrees)
            
            # Position the rotated car
            car_rect = rotated_surface.get_rect()
            car_rect.center = (screen_x, screen_y)
            
            # Draw shadow first
            shadow_rect = car_rect.copy()
            shadow_rect.center = (screen_x + 2, screen_y + 2)
            shadow_surface = rotated_surface.copy()
            shadow_surface.fill((0, 0, 0, 60), special_flags=pygame.BLEND_RGBA_MULT)
            self.screen.blit(shadow_surface, shadow_rect)
            
            # Draw main car
            self.screen.blit(rotated_surface, car_rect)
        
        # Draw speed multiplier effect with enhanced visuals
        if hasattr(car, 'speed_multiplier') and car.speed_multiplier != 1.0:
            effect_radius = 20
            if car.speed_multiplier > 1.0:
                # Green boost effect with particles
                effect_color = (0, 255, 0)
                particle_color = (100, 255, 100)
                # Draw pulsing effect
                pulse_radius = int(effect_radius + 5 * math.sin(pygame.time.get_ticks() * 0.01))
                pygame.draw.circle(self.screen, (*effect_color, 100), 
                                 (screen_x, screen_y), pulse_radius, 3)
                # Add particle effect
                for i in range(3):
                    offset_angle = car.angle + i * 2.1
                    px = int(screen_x + 15 * math.cos(offset_angle))
                    py = int(screen_y - 15 * math.sin(offset_angle))
                    pygame.draw.circle(self.screen, particle_color, (px, py), 2)
            else:
                # Red slowdown effect
                effect_color = (255, 0, 0)
                pygame.draw.circle(self.screen, (*effect_color, 120), 
                                 (screen_x, screen_y), effect_radius, 4)
                # Add warning indicators
                for i in range(4):
                    angle = i * math.pi / 2
                    px = int(screen_x + 25 * math.cos(angle))
                    py = int(screen_y + 25 * math.sin(angle))
                    pygame.draw.polygon(self.screen, effect_color, 
                                      [(px, py-3), (px-3, py+3), (px+3, py+3)])
        
        # Draw enhanced label with background
        if label:
            text = self.small_font.render(label, True, WHITE)
            text_rect = text.get_rect(center=(screen_x, screen_y - 25))
            
            # Background for label
            bg_rect = text_rect.inflate(6, 2)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), bg_rect, border_radius=3)
            pygame.draw.rect(self.screen, WHITE, bg_rect, 1, border_radius=3)
            
            self.screen.blit(text, text_rect)
                
    def render_cars(self, cars: List[Car]):
        """Render all cars with enhanced graphics"""
        for i, car in enumerate(cars):
            if isinstance(car, BaselineCar):
                self.render_car(car, "baseline", "BASE")
            elif isinstance(car, RLCar):
                self.render_car(car, "dqn", "RL")
            else:
                self.render_car(car, "default", f"CAR{i}")
                
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
        """Main render function for circular track with enhanced graphics"""
        # Draw grass background
        self.screen.blit(self.grass_surface, (0, 0))
        
        # Render track and game components
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