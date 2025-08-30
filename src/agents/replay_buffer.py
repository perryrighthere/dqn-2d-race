import random
import numpy as np
from collections import deque, namedtuple
from typing import Tuple, List, Optional
import torch

# Define experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    """Experience replay buffer for DQN training"""
    
    def __init__(self, buffer_size: int = 100000, batch_size: int = 32, seed: int = 42):
        """Initialize replay buffer
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Size of each training batch
            seed: Random seed for reproducibility
        """
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, ...]:
        """Sample random batch of experiences from buffer
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        if len(self.buffer) < self.batch_size:
            raise ValueError(f"Buffer contains only {len(self.buffer)} experiences, "
                           f"need at least {self.batch_size} for sampling")
        
        experiences = random.sample(self.buffer, self.batch_size)
        
        # Convert experiences to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.tensor([bool(e.done) for e in experiences], dtype=torch.bool).to(device)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer"""
        return len(self.buffer)
    
    def can_sample(self) -> bool:
        """Check if buffer has enough experiences for sampling"""
        return len(self.buffer) >= self.batch_size
    
    def clear(self):
        """Clear all experiences from buffer"""
        self.buffer.clear()
    
    def get_stats(self) -> dict:
        """Get buffer statistics"""
        if len(self.buffer) == 0:
            return {"size": 0, "utilization": 0.0, "avg_reward": 0.0}
            
        rewards = [exp.reward for exp in self.buffer]
        return {
            "size": len(self.buffer),
            "utilization": len(self.buffer) / self.buffer_size,
            "avg_reward": np.mean(rewards),
            "min_reward": np.min(rewards),
            "max_reward": np.max(rewards),
            "buffer_size": self.buffer_size
        }

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer for improved DQN training"""
    
    def __init__(self, buffer_size: int = 100000, batch_size: int = 32, 
                 alpha: float = 0.6, beta: float = 0.4, seed: int = 42):
        """Initialize prioritized replay buffer
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Size of each training batch
            alpha: Prioritization exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent (corrects bias from prioritization)
            seed: Random seed for reproducibility
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6  # Small constant to avoid zero priorities
        
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool, priority: Optional[float] = None):
        """Add experience to buffer with priority
        
        Args:
            priority: TD error for prioritization (if None, uses max priority)
        """
        experience = Experience(state, action, reward, next_state, done)
        
        # Use maximum priority for new experiences if not specified
        if priority is None:
            priority = max(self.priorities) if self.priorities else 1.0
            
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor, ...]:
        """Sample batch based on priorities
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, indices, weights)
        """
        if len(self.buffer) < self.batch_size:
            raise ValueError(f"Buffer contains only {len(self.buffer)} experiences")
            
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float64)
        priorities = priorities ** self.alpha
        probabilities = priorities / np.sum(priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), size=self.batch_size, 
                                 p=probabilities, replace=False)
        
        # Get experiences
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / np.max(weights)  # Normalize weights
        
        # Convert to tensors
        states = torch.FloatTensor(np.array([e.state for e in experiences])).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).to(device)
        next_states = torch.FloatTensor(np.array([e.next_state for e in experiences])).to(device)
        dones = torch.tensor([bool(e.done) for e in experiences], dtype=torch.bool).to(device)
        weights = torch.FloatTensor(weights).to(device)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities for given experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def can_sample(self) -> bool:
        return len(self.buffer) >= self.batch_size
    
    def get_stats(self) -> dict:
        """Get buffer statistics including priority information"""
        if len(self.buffer) == 0:
            return {"size": 0, "utilization": 0.0}
            
        rewards = [exp.reward for exp in self.buffer]
        priorities = list(self.priorities)
        
        return {
            "size": len(self.buffer),
            "utilization": len(self.buffer) / self.buffer_size,
            "avg_reward": np.mean(rewards),
            "avg_priority": np.mean(priorities),
            "max_priority": np.max(priorities),
            "min_priority": np.min(priorities)
        }

def create_replay_buffer(buffer_type: str = "standard", **kwargs) -> ReplayBuffer:
    """Factory function to create replay buffers
    
    Args:
        buffer_type: "standard" or "prioritized"
        **kwargs: Additional arguments for buffer construction
        
    Returns:
        Replay buffer instance
    """
    if buffer_type.lower() == "standard":
        return ReplayBuffer(**kwargs)
    elif buffer_type.lower() == "prioritized":
        return PrioritizedReplayBuffer(**kwargs)
    else:
        raise ValueError(f"Unknown buffer type: {buffer_type}. Use 'standard' or 'prioritized'")

def test_replay_buffer():
    """Test replay buffer functionality"""
    print("Testing Replay Buffer...")
    
    # Test standard replay buffer
    buffer = ReplayBuffer(buffer_size=1000, batch_size=4)
    
    # Add some experiences
    for i in range(10):
        state = np.random.randn(11)
        action = np.random.randint(5)
        reward = np.random.randn()
        next_state = np.random.randn(11)
        done = np.random.choice([True, False])
        
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Can sample: {buffer.can_sample()}")
    print(f"Stats: {buffer.get_stats()}")
    
    # Test sampling
    if buffer.can_sample():
        states, actions, rewards, next_states, dones = buffer.sample()
        print(f"Sample shapes: states={states.shape}, actions={actions.shape}")
        print(f"Sample dtypes: states={states.dtype}, actions={actions.dtype}")
    
    # Test prioritized replay buffer
    print("\nTesting Prioritized Replay Buffer...")
    pri_buffer = PrioritizedReplayBuffer(buffer_size=1000, batch_size=4)
    
    # Add experiences with different priorities
    for i in range(10):
        state = np.random.randn(11)
        action = np.random.randint(5)
        reward = np.random.randn()
        next_state = np.random.randn(11)
        done = np.random.choice([True, False])
        priority = np.random.rand()  # Random priority
        
        pri_buffer.add(state, action, reward, next_state, done, priority)
    
    print(f"Prioritized buffer stats: {pri_buffer.get_stats()}")
    
    if pri_buffer.can_sample():
        states, actions, rewards, next_states, dones, indices, weights = pri_buffer.sample()
        print(f"Prioritized sample shapes: weights={weights.shape}")
        
        # Test priority updates
        new_priorities = np.random.rand(len(indices))
        pri_buffer.update_priorities(indices, new_priorities)
        print("Priority updates completed")
    
    # Test factory function
    std_buffer = create_replay_buffer("standard", buffer_size=500)
    pri_buffer = create_replay_buffer("prioritized", buffer_size=500, alpha=0.7)
    
    print(f"Factory created buffers: {type(std_buffer).__name__}, {type(pri_buffer).__name__}")
    print("All replay buffer tests passed!")

if __name__ == "__main__":
    test_replay_buffer()