import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, Dict, List, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from .dqn_network import DoubleDQNNetwork, create_dqn_network
from .replay_buffer import ReplayBuffer, create_replay_buffer

class DQNAgent:
    """Deep Q-Network Agent for 2D Car Racing"""
    
    def __init__(self, state_size: int = 11, action_size: int = 5,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 100000,
                 batch_size: int = 64, update_freq: int = 4,
                 target_update_freq: int = 100, device: str = "auto",
                 network_type: str = "double", buffer_type: str = "standard",
                 hidden_layers: List[int] = None, seed: int = 42):
        """Initialize DQN Agent
        
        Args:
            state_size: Size of the state space
            action_size: Number of possible actions  
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of replay buffer
            batch_size: Batch size for training
            update_freq: Frequency of training updates (in steps)
            target_update_freq: Frequency of target network updates
            device: Computing device ("cpu", "cuda", or "auto")
            network_type: "simple" or "double" DQN
            buffer_type: "standard" or "prioritized" replay buffer
            hidden_layers: List of hidden layer sizes (e.g., [128, 128, 64])
            seed: Random seed for reproducibility
        """
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"DQN Agent using device: {self.device}")
        
        # Agent parameters
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        
        # Training state
        self.step_count = 0
        self.episode_count = 0
        self.total_reward = 0.0
        self.losses = []
        self.rewards_history = []
        
        # Create networks
        self.network_type = network_type
        if network_type == "double":
            self.q_network = create_dqn_network(
                state_size, action_size, network_type="double", hidden_layers=hidden_layers
            ).to(self.device)
            self.optimizer = optim.Adam(self.q_network.online_network.parameters(), 
                                      lr=learning_rate)
        else:
            self.q_network = create_dqn_network(
                state_size, action_size, network_type="simple", hidden_layers=hidden_layers
            ).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Create replay buffer
        self.replay_buffer = create_replay_buffer(
            buffer_type=buffer_type,
            buffer_size=buffer_size,
            batch_size=batch_size,
            seed=seed
        )
        
        print(f"DQN Agent initialized:")
        print(f"  Network: {network_type.title()} DQN")
        print(f"  Buffer: {buffer_type.title()} Replay Buffer")
        print(f"  State size: {state_size}, Action size: {action_size}")
        
    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Choose action using epsilon-greedy policy"""
        # Use current epsilon for exploration during training
        current_epsilon = self.epsilon if training else 0.0
        
        if self.network_type == "double":
            return self.q_network.get_action(state, current_epsilon)
        else:
            return self.q_network.get_action(state, current_epsilon)
    
    def step(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """Store experience and train if ready"""
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Update step count
        self.step_count += 1
        
        # Train if we have enough experiences and it's time to update
        if (self.replay_buffer.can_sample() and 
            self.step_count % self.update_freq == 0):
            self.learn()
        
        # Update target network periodically (for double DQN)
        if (self.network_type == "double" and 
            self.step_count % self.target_update_freq == 0):
            self.q_network.update_target_network()
    
    def learn(self):
        """Train the agent using a batch of experiences"""
        try:
            # Sample batch from replay buffer
            if hasattr(self.replay_buffer, 'sample'):
                batch = self.replay_buffer.sample(self.device)
                if len(batch) == 5:  # Standard replay buffer
                    states, actions, rewards, next_states, dones = batch
                    weights = None
                else:  # Prioritized replay buffer
                    states, actions, rewards, next_states, dones, indices, weights = batch
            else:
                return  # Skip if buffer doesn't support sampling
                
            # Compute current Q values
            if self.network_type == "double":
                current_q_values = self.q_network(states, use_target=False)
            else:
                current_q_values = self.q_network(states)
            
            current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Compute next Q values
            with torch.no_grad():
                if self.network_type == "double":
                    # Double DQN: use online network to select actions, target to evaluate
                    next_actions = self.q_network(next_states, use_target=False).argmax(1)
                    next_q_values = self.q_network(next_states, use_target=True)
                    next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                else:
                    # Standard DQN
                    next_q_values = self.q_network(next_states).max(1)[0]
                
                # Calculate target Q values
                target_q_values = rewards + (self.gamma * next_q_values * (~dones))
            
            # Compute loss
            if weights is not None:
                # Weighted loss for prioritized experience replay
                loss = (weights * nn.MSELoss(reduction='none')(current_q_values, target_q_values)).mean()
                
                # Update priorities if using prioritized replay
                td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
                self.replay_buffer.update_priorities(indices, td_errors)
            else:
                loss = nn.MSELoss()(current_q_values, target_q_values)
            
            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            if self.network_type == "double":
                torch.nn.utils.clip_grad_norm_(self.q_network.online_network.parameters(), 1.0)
            else:
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
                
            self.optimizer.step()
            
            # Store loss for monitoring
            self.losses.append(loss.item())
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        
        except Exception as e:
            print(f"Error during learning: {e}")
            import traceback
            traceback.print_exc()
    
    def end_episode(self, total_reward: float):
        """Called at the end of each episode"""
        self.episode_count += 1
        self.rewards_history.append(total_reward)
        
        # Print progress periodically
        if self.episode_count % 10 == 0:
            avg_reward = np.mean(self.rewards_history[-10:])
            avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
            buffer_util = len(self.replay_buffer) / self.replay_buffer.buffer_size
            
            print(f"Episode {self.episode_count:4d} | "
                  f"Avg Reward: {avg_reward:7.2f} | "
                  f"Epsilon: {self.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buffer: {buffer_util:.1%}")
    
    def get_stats(self) -> Dict:
        """Get training statistics"""
        recent_rewards = self.rewards_history[-100:] if self.rewards_history else [0]
        recent_losses = self.losses[-100:] if self.losses else [0]
        
        return {
            "episode": self.episode_count,
            "step": self.step_count,
            "epsilon": self.epsilon,
            "avg_reward_100": np.mean(recent_rewards),
            "avg_loss_100": np.mean(recent_losses),
            "buffer_size": len(self.replay_buffer),
            "buffer_utilization": len(self.replay_buffer) / self.replay_buffer.buffer_size,
            "total_episodes": len(self.rewards_history)
        }
    
    def save_model(self, filepath: str):
        """Save model weights and training state"""
        checkpoint = {
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'epsilon': self.epsilon,
            'rewards_history': self.rewards_history,
            'losses': self.losses[-1000:],  # Save last 1000 losses
            'network_type': self.network_type,
            'state_size': self.state_size,
            'action_size': self.action_size,
        }
        
        if self.network_type == "double":
            checkpoint['online_network_state_dict'] = self.q_network.online_network.state_dict()
            checkpoint['target_network_state_dict'] = self.q_network.target_network.state_dict()
        else:
            checkpoint['network_state_dict'] = self.q_network.state_dict()
        
        checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str, load_optimizer: bool = True):
        """Load model weights and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network weights
        if self.network_type == "double":
            self.q_network.online_network.load_state_dict(checkpoint['online_network_state_dict'])
            self.q_network.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        else:
            self.q_network.load_state_dict(checkpoint['network_state_dict'])
        
        # Load optimizer state
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.step_count = checkpoint.get('step_count', 0)
        self.episode_count = checkpoint.get('episode_count', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.rewards_history = checkpoint.get('rewards_history', [])
        self.losses = checkpoint.get('losses', [])
        
        print(f"Model loaded from {filepath}")
        print(f"Resumed at episode {self.episode_count}, step {self.step_count}")

def create_racing_dqn_agent(config: Dict = None) -> DQNAgent:
    """Factory function to create DQN agent with racing-optimized hyperparameters"""
    
    default_config = {
        'state_size': 11,
        'action_size': 5,
        'learning_rate': 0.0005,  # Slightly lower for stability
        'gamma': 0.99,           # Standard discount factor
        'epsilon': 1.0,          # Start with full exploration
        'epsilon_min': 0.01,     # Allow some exploration
        'epsilon_decay': 0.9995, # Gradual decay
        'buffer_size': 50000,    # Moderate buffer size
        'batch_size': 64,        # Good batch size for training
        'update_freq': 4,        # Update every 4 steps
        'target_update_freq': 1000,  # Update target every 1000 steps
        'network_type': 'double',    # Use double DQN
        'buffer_type': 'standard',   # Standard replay buffer
        'hidden_layers': [128, 128, 64],  # Default network architecture
        'device': 'auto',
        'seed': 42
    }
    
    if config:
        default_config.update(config)
    
    return DQNAgent(**default_config)

def test_dqn_agent():
    """Test DQN agent functionality"""
    print("Testing DQN Agent...")
    
    # Create agent
    agent = create_racing_dqn_agent()
    
    # Test action selection
    test_state = np.random.randn(11)
    action = agent.act(test_state)
    print(f"Selected action: {action}")
    
    # Test experience storage and learning
    for i in range(100):
        state = np.random.randn(11)
        action = np.random.randint(5)
        reward = np.random.randn()
        next_state = np.random.randn(11)
        done = i % 20 == 0  # Episode ends every 20 steps
        
        agent.step(state, action, reward, next_state, done)
        
        if done:
            agent.end_episode(reward * 10)
    
    # Test statistics
    stats = agent.get_stats()
    print(f"Training stats: {stats}")
    
    # Test save/load
    agent.save_model("test_dqn_model.pth")
    
    # Create new agent and load
    new_agent = create_racing_dqn_agent()
    new_agent.load_model("test_dqn_model.pth")
    
    print("Model save/load test completed")
    
    # Clean up test file
    import os
    if os.path.exists("test_dqn_model.pth"):
        os.remove("test_dqn_model.pth")
    
    print("All DQN agent tests passed!")

if __name__ == "__main__":
    test_dqn_agent()