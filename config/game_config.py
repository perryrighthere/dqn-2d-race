# Game configuration parameters

# Track settings - Circular Track
TRACK_RADIUS = 300   # Radius of the circular track center
TRACK_WIDTH = 120    # Width of the track (distance between inner and outer edges)
NUM_LANES = 3        # Number of lanes
LANE_WIDTH = TRACK_WIDTH // NUM_LANES
LAPS_TO_WIN = 3      # Number of laps required to win the race

# Car settings
CAR_WIDTH = 30
CAR_HEIGHT = 20
BASE_SPEED = 5.0     # Baseline agent speed
MAX_SPEED = 10.0
MIN_SPEED = 1.0

# Special tiles
ACCELERATION_BOOST = 1.5  # Speed multiplier for acceleration tiles
DECELERATION_FACTOR = 0.5  # Speed multiplier for deceleration tiles
TILE_SIZE = 50       # Size of special tiles
TILE_DENSITY = 0.1   # Probability of tile placement per unit distance

# Display settings
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colors (RGB)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (128, 128, 128)