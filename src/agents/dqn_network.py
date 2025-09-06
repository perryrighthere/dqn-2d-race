import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List

class DQNNetwork(nn.Module):
    """Deep Q-Network for 2D Car Racing Agent with configurable architecture"""
    
    def __init__(self, state_size: int = 11, action_size: int = 5, 
                 hidden_layers: List[int] = None, dropout_rate: float = 0.1):
        super(DQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Default architecture if none provided
        if hidden_layers is None:
            hidden_layers = [128, 128, 64]
        
        self.hidden_layers = hidden_layers
        
        # Build configurable network architecture
        layers = []
        dropouts = []
        
        # Input layer
        prev_size = state_size
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        self.hidden_nets = nn.ModuleList(layers)
        self.dropout_layers = nn.ModuleList(dropouts)
        
        # Dueling DQN architecture - use last hidden layer size
        final_hidden_size = hidden_layers[-1]
        
        # Value stream - estimates state value
        self.value_stream = nn.Linear(final_hidden_size, 1)
        
        # Advantage stream - estimates action advantages
        self.advantage_stream = nn.Linear(final_hidden_size, action_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize network weights using Xavier initialization"""
        # Initialize hidden layers
        for layer in self.hidden_nets:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        
        # Initialize dueling streams
        for layer in [self.value_stream, self.advantage_stream]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure input is float32
        if state.dtype != torch.float32:
            state = state.float()
            
        # Forward pass through configurable hidden layers
        x = state
        for i, (layer, dropout) in enumerate(zip(self.hidden_nets, self.dropout_layers)):
            x = F.relu(layer(x))
            x = dropout(x)
        
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
                 hidden_layers: List[int] = None, dropout_rate: float = 0.1):
        super(DoubleDQNNetwork, self).__init__()
        
        # Default architecture if none provided
        if hidden_layers is None:
            hidden_layers = [128, 128, 64]
        
        # Main network (online network)
        self.online_network = DQNNetwork(state_size, action_size, hidden_layers, dropout_rate)
        
        # Target network (for stable training)
        self.target_network = DQNNetwork(state_size, action_size, hidden_layers, dropout_rate)
        
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