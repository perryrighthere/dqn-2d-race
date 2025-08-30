import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class DQNNetwork(nn.Module):
    """Deep Q-Network for 2D Car Racing Agent"""
    
    def __init__(self, state_size: int = 11, action_size: int = 5, 
                 hidden_size: int = 128, dropout_rate: float = 0.1):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Network architecture optimized for racing task
        # Input: state_size (11) -> car state (5) + tile info (6)
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Dueling DQN architecture
        # Value stream - estimates state value
        self.value_stream = nn.Linear(hidden_size // 2, 1)
        
        # Advantage stream - estimates action advantages
        self.advantage_stream = nn.Linear(hidden_size // 2, action_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        for layer in [self.fc1, self.fc2, self.fc3, self.value_stream, self.advantage_stream]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure input is float32
        if state.dtype != torch.float32:
            state = state.float()
            
        # Feature extraction layers
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        # Dueling DQN: split into value and advantage streams
        value = self.value_stream(x)  # Shape: (batch_size, 1)
        advantage = self.advantage_stream(x)  # Shape: (batch_size, action_size)
        
        # Combine value and advantage to get Q-values
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
        
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state)
            return q_values.argmax().item()
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state"""
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.forward(state)
            return q_values.squeeze().cpu().numpy()

class DoubleDQNNetwork(nn.Module):
    """Double DQN implementation with separate target network"""
    
    def __init__(self, state_size: int = 11, action_size: int = 5, 
                 hidden_size: int = 128, dropout_rate: float = 0.1):
        super(DoubleDQNNetwork, self).__init__()
        
        # Main network (online network)
        self.online_network = DQNNetwork(state_size, action_size, hidden_size, dropout_rate)
        
        # Target network (for stable training)
        self.target_network = DQNNetwork(state_size, action_size, hidden_size, dropout_rate)
        
        # Copy weights from online to target network
        self.update_target_network()
        
        # Freeze target network
        for param in self.target_network.parameters():
            param.requires_grad = False
    
    def forward(self, state: torch.Tensor, use_target: bool = False) -> torch.Tensor:
        """Forward pass through online or target network"""
        if use_target:
            return self.target_network(state)
        else:
            return self.online_network(state)
    
    def update_target_network(self, tau: float = 1.0):
        """Update target network weights
        
        Args:
            tau: Soft update parameter (1.0 = hard update, <1.0 = soft update)
        """
        for target_param, online_param in zip(self.target_network.parameters(), 
                                            self.online_network.parameters()):
            target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)
    
    def get_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """Select action using online network with epsilon-greedy policy"""
        return self.online_network.get_action(state, epsilon)
    
    def get_q_values(self, state: np.ndarray, use_target: bool = False) -> np.ndarray:
        """Get Q-values using online or target network"""
        if use_target:
            return self.target_network.get_q_values(state)
        else:
            return self.online_network.get_q_values(state)

def create_dqn_network(state_size: int = 11, action_size: int = 5, 
                      network_type: str = "double", **kwargs) -> nn.Module:
    """Factory function to create DQN networks
    
    Args:
        state_size: Size of the state space
        action_size: Number of possible actions
        network_type: "simple" or "double" DQN
        **kwargs: Additional arguments for network construction
    
    Returns:
        DQN network instance
    """
    if network_type.lower() == "double":
        return DoubleDQNNetwork(state_size, action_size, **kwargs)
    elif network_type.lower() == "simple":
        return DQNNetwork(state_size, action_size, **kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}. Use 'simple' or 'double'")

def test_network():
    """Test DQN network functionality"""
    print("Testing DQN Network...")
    
    # Test simple DQN
    simple_dqn = DQNNetwork()
    test_state = torch.randn(1, 11)  # Batch of 1, state size 11
    q_values = simple_dqn(test_state)
    print(f"Simple DQN output shape: {q_values.shape}")  # Should be (1, 5)
    
    # Test action selection
    state_np = np.random.randn(11)
    action = simple_dqn.get_action(state_np, epsilon=0.1)
    print(f"Selected action: {action}")
    
    # Test Double DQN
    double_dqn = DoubleDQNNetwork()
    online_q = double_dqn(test_state, use_target=False)
    target_q = double_dqn(test_state, use_target=True)
    print(f"Double DQN online output shape: {online_q.shape}")
    print(f"Double DQN target output shape: {target_q.shape}")
    
    # Test target network update
    double_dqn.update_target_network(tau=0.01)
    print("Target network updated successfully")
    
    # Test factory function
    network = create_dqn_network(network_type="double", hidden_size=64)
    print(f"Factory created network type: {type(network).__name__}")
    
    print("All network tests passed!")

if __name__ == "__main__":
    test_network()