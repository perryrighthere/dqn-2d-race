"""
Simple random agent for testing purposes
"""
import numpy as np
from gymnasium.spaces import Space

class RandomAgent:
    """Agent that takes random actions"""
    
    def __init__(self, action_space: Space):
        """Initialize random agent with action space"""
        self.action_space = action_space
        
    def act(self, observation) -> int:
        """Take a random action"""
        return self.action_space.sample()
        
    def reset(self):
        """Reset agent state (no-op for random agent)"""
        pass