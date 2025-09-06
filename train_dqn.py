#!/usr/bin/env python3
"""
Phase 3: DQN Agent Training
Train DQN agent to compete against baseline agent in 2D car racing
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.environment.race_environment import RaceEnvironment
from src.agents.dqn_agent import create_racing_dqn_agent, DQNAgent
from glob import glob

class TrainingManager:
    """Manages DQN training process and logging"""
    
    def __init__(self, config: Dict = None):
        """Initialize training manager
        
        Args:
            config: Training configuration dictionary
        """
        # Default training configuration
        self.config = {
            # Training parameters
            'num_episodes': 1000,
            'max_steps_per_episode': 4200,  # 70 seconds at 60 FPS
            'save_freq': 100,  # Save model every 100 episodes
            'eval_freq': 50,   # Evaluate every 50 episodes
            'eval_episodes': 10,  # Number of episodes for evaluation
            'render_eval': False,  # Whether to render during evaluation
            
            # Environment parameters
            'render_training': False,
            'seed': 42,
            
            # Agent parameters - optimized for racing
            'agent_config': {
                'learning_rate': 0.0005,
                'gamma': 0.99,
                'epsilon': 1.0,
                'epsilon_min': 0.05,  # Keep some exploration
                'epsilon_decay': 0.9995,
                'buffer_size': 100000,
                'batch_size': 64,
                'update_freq': 4,
                'target_update_freq': 1000,
                'network_type': 'double',
                'buffer_type': 'standard'
            },
            
            # Evaluation parameters
            'eval_tile_density': None,  # Use this tile density during eval if set
            
            # Logging and saving
            'save_dir': 'models',
            'log_dir': 'logs',
            'plot_training': True,

            # Curriculum learning (optional)
            'curriculum': {
                'enabled': False,
                'stages': [],  # e.g., [{'start_episode':1,'tile_density':0.5}, ...]
                'adaptive': True,
                'advance_win_rate': 0.85,
                'window': 50
            }
        }
        
        if config:
            self._update_config(self.config, config)
        
        # Create directories
        os.makedirs(self.config['save_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Training history
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'wins': [],
            'losses': [],
            'epsilons': [],
            'race_times': [],
            'tile_density': [],
            'evaluation_results': []
        }
        
        # Initialize environment and agent
        self.env = None
        self.agent = None
        self.start_time = None
        
    def _update_config(self, base_config: Dict, update_config: Dict):
        """Recursively update configuration"""
        for key, value in update_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def setup(self):
        """Setup environment and agent"""
        print("Setting up training environment...")
        
        # Create environment
        render_mode = "human" if self.config['render_training'] else None
        self.env = RaceEnvironment(render_mode=render_mode)
        
        # Set seed for reproducibility
        if self.config['seed'] is not None:
            np.random.seed(self.config['seed'])
        
        # Create DQN agent
        agent_config = self.config['agent_config'].copy()
        agent_config['seed'] = self.config['seed']
        self.agent = create_racing_dqn_agent(agent_config)
        
        print(f"Training setup completed:")
        print(f"  Episodes: {self.config['num_episodes']}")
        print(f"  Max steps per episode: {self.config['max_steps_per_episode']}")
        print(f"  Agent: {self.agent.network_type.title()} DQN")
        print(f"  Buffer: {type(self.agent.replay_buffer).__name__}")
        
    def train_episode(self, episode: int) -> Tuple[float, bool, Dict]:
        """Train for one episode"""
        # Determine curriculum tile density for this episode
        tile_density = self._get_tile_density_for_episode(episode)
        state, info = self.env.reset(seed=self.config['seed'] + episode,
                                     options={'tile_density': tile_density})
        total_reward = 0.0
        steps = 0
        episode_info = {
            'race_time': 0.0,
            'rl_laps': 0,
            'baseline_laps': 0,
            'winner': None
        }
        
        for step in range(self.config['max_steps_per_episode']):
            # Agent selects action
            action = self.agent.act(state, training=True)
            
            # Environment step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Agent learns from experience
            self.agent.step(state, action, reward, next_state, terminated or truncated)
            
            # Update state and reward
            state = next_state
            total_reward += reward
            steps += 1
            
            # Store episode info
            episode_info.update({
                'race_time': info['race_time'],
                'rl_laps': info['rl_car_laps'],
                'baseline_laps': info['baseline_car_laps'],
                'winner': info.get('winner')
            })
            
            # Check if episode is done
            if terminated or truncated:
                break
        
        # End episode for agent
        self.agent.end_episode(total_reward)
        
        # Determine if RL agent won
        won = episode_info['winner'] == 'RL'
        
        return total_reward, won, episode_info
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """Evaluate agent performance"""
        print(f"Evaluating agent for {num_episodes} episodes...")
        
        eval_rewards = []
        eval_wins = []
        eval_race_times = []
        
        eval_tile_density = self.config.get('eval_tile_density')
        for episode in range(num_episodes):
            state, info = self.env.reset(seed=1000 + episode,
                                         options={'tile_density': eval_tile_density})  # Different seed for eval
            total_reward = 0.0
            
            for step in range(self.config['max_steps_per_episode']):
                # Agent acts without exploration
                action = self.agent.act(state, training=False)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                
                state = next_state
                total_reward += reward
                
                if terminated or truncated:
                    break
            
            # Record results
            eval_rewards.append(total_reward)
            eval_wins.append(info.get('winner') == 'RL')
            eval_race_times.append(info['race_time'])
        
        # Determine baseline time (load latest measured mean if available)
        def _load_baseline_time(default: float = 18.83) -> float:
            try:
                files = sorted(glob("baseline_benchmark_*.json"))
                if files:
                    latest = files[-1]
                    with open(latest, 'r') as f:
                        data = json.load(f)
                    stats = data.get('statistical_analysis', {}).get('completion_time_stats', {})
                    mean_time = stats.get('mean')
                    if isinstance(mean_time, (int, float)) and mean_time > 0:
                        return float(mean_time)
            except Exception:
                pass
            return default

        baseline_time_val = _load_baseline_time()

        # Calculate statistics
        eval_results = {
            'avg_reward': np.mean(eval_rewards),
            'win_rate': np.mean(eval_wins),
            'avg_race_time': np.mean(eval_race_times),
            'baseline_time': baseline_time_val,
            'time_improvement': baseline_time_val - np.mean(eval_race_times)
        }
        
        print(f"Evaluation Results:")
        print(f"  Average Reward: {eval_results['avg_reward']:.2f}")
        print(f"  Win Rate: {eval_results['win_rate']:.1%}")
        print(f"  Average Race Time: {eval_results['avg_race_time']:.2f}s")
        print(f"  Time vs Baseline: {eval_results['time_improvement']:+.2f}s (baseline {baseline_time_val:.2f}s)")
        
        return eval_results
    
    def train(self):
        """Main training loop"""
        print("Starting DQN training...")
        self.start_time = time.time()
        
        for episode in range(1, self.config['num_episodes'] + 1):
            # Train one episode
            reward, won, episode_info = self.train_episode(episode)
            
            # Store training history
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(reward)
            self.training_history['wins'].append(won)
            self.training_history['race_times'].append(episode_info['race_time'])
            self.training_history['epsilons'].append(self.agent.epsilon)
            # Record tile density used (None means default)
            # Note: last reset's tile density is stored in env.info as well
            try:
                self.training_history['tile_density'].append(self.env.tile_density)
            except Exception:
                self.training_history['tile_density'].append(None)
            
            # Store recent loss
            recent_losses = self.agent.losses[-10:] if self.agent.losses else [0]
            self.training_history['losses'].append(np.mean(recent_losses))
            
            # Periodic evaluation
            if episode % self.config['eval_freq'] == 0:
                eval_results = self.evaluate(self.config['eval_episodes'])
                eval_results['episode'] = episode
                self.training_history['evaluation_results'].append(eval_results)
            
            # Save model periodically
            if episode % self.config['save_freq'] == 0:
                model_path = os.path.join(self.config['save_dir'], 
                                        f'dqn_racing_episode_{episode}.pth')
                self.agent.save_model(model_path)
                
                # Also save training history
                self.save_training_history(episode)
            
            # Print progress
            if episode % 10 == 0:
                elapsed_time = time.time() - self.start_time
                avg_reward = np.mean(self.training_history['rewards'][-50:])
                win_rate = np.mean(self.training_history['wins'][-50:])
                
                print(f"Episode {episode:4d}/{self.config['num_episodes']} | "
                      f"Reward: {avg_reward:7.2f} | "
                      f"Win Rate: {win_rate:.1%} | "
                      f"Epsilon: {self.agent.epsilon:.3f} | "
                      f"Time: {elapsed_time:.0f}s")
        
        # Final evaluation and save
        print("\nTraining completed! Running final evaluation...")
        final_eval = self.evaluate(20)  # More episodes for final evaluation
        final_eval['episode'] = self.config['num_episodes']
        self.training_history['evaluation_results'].append(final_eval)
        
        # Save final model and history
        final_model_path = os.path.join(self.config['save_dir'], 'dqn_racing_final.pth')
        self.agent.save_model(final_model_path)
        self.save_training_history('final')
        
        # Generate plots
        if self.config['plot_training']:
            self.plot_training_results()
        
        print(f"Training completed in {time.time() - self.start_time:.0f} seconds")
        return final_eval
    
    def save_training_history(self, suffix: str):
        """Save training history to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"training_history_{suffix}_{timestamp}.json"
        filepath = os.path.join(self.config['log_dir'], filename)
        
        # Convert numpy types to native Python types for JSON serialization
        history = {}
        for key, value in self.training_history.items():
            if isinstance(value, list):
                history[key] = [float(v) if isinstance(v, (np.float32, np.float64)) 
                              else bool(v) if isinstance(v, (np.bool_, bool)) 
                              else v for v in value]
            else:
                history[key] = value
        
        with open(filepath, 'w') as f:
            json.dump({
                'config': self.config,
                'history': history,
                'final_stats': self.agent.get_stats()
            }, f, indent=2)
        
        print(f"Training history saved to {filepath}")
    
    def plot_training_results(self):
        """Generate training plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = self.training_history['episodes']
        
        # Reward plot
        axes[0, 0].plot(episodes, self.training_history['rewards'], alpha=0.3)
        # Moving average
        window = min(50, len(episodes) // 10)
        if window > 1:
            rewards_smooth = np.convolve(self.training_history['rewards'], 
                                       np.ones(window)/window, mode='valid')
            axes[0, 0].plot(episodes[window-1:], rewards_smooth, 'r-', linewidth=2)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].grid(True)
        
        # Win rate plot
        window = min(20, len(episodes) // 5)
        if window > 1:
            wins_smooth = np.convolve(self.training_history['wins'], 
                                    np.ones(window)/window, mode='valid')
            axes[0, 1].plot(episodes[window-1:], wins_smooth, 'g-', linewidth=2)
        axes[0, 1].set_title('Win Rate (Moving Average)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True)
        
        # Epsilon decay
        axes[1, 0].plot(episodes, self.training_history['epsilons'], 'b-')
        axes[1, 0].set_title('Epsilon Decay')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].grid(True)
        
        # Loss plot
        if len(self.training_history['losses']) > 10:
            axes[1, 1].plot(episodes, self.training_history['losses'], alpha=0.5)
            window = min(20, len(episodes) // 10)
            if window > 1:
                loss_smooth = np.convolve(self.training_history['losses'], 
                                        np.ones(window)/window, mode='valid')
                axes[1, 1].plot(episodes[window-1:], loss_smooth, 'r-', linewidth=2)
        axes[1, 1].set_title('Training Loss')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config['log_dir'], 
                               f"training_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training plots saved to {plot_path}")
    
    def cleanup(self):
        """Clean up resources"""
        if self.env:
            self.env.close()

    def _get_tile_density_for_episode(self, episode: int) -> float:
        """Return tile density for current episode based on curriculum config"""
        curriculum = self.config.get('curriculum', {}) or {}
        if not curriculum.get('enabled'):
            return None  # Use environment default
        stages = sorted(curriculum.get('stages', []), key=lambda s: s.get('start_episode', 1))
        if not stages:
            return None
        # Determine stage by episode
        current_density = stages[0].get('tile_density')
        for st in stages:
            if episode >= st.get('start_episode', 1):
                current_density = st.get('tile_density', current_density)
            else:
                break
        # Simple adaptive advancement: if recent win rate is high, use next stage density
        if curriculum.get('adaptive') and len(self.training_history['wins']) >= curriculum.get('window', 50):
            window = curriculum.get('window', 50)
            win_rate_recent = np.mean(self.training_history['wins'][-window:])
            if win_rate_recent >= curriculum.get('advance_win_rate', 0.85):
                for st in stages:
                    if st.get('start_episode', 1) > episode:
                        current_density = st.get('tile_density', current_density)
                        break
        return current_density

def create_training_config(config_type: str = "standard") -> Dict:
    """Create predefined training configurations"""
    
    configs = {
        "fast": {  # Quick training for testing
            'num_episodes': 200,
            'save_freq': 50,
            'eval_freq': 25,
            'agent_config': {
                'buffer_size': 10000,
                'batch_size': 32,
                'epsilon_decay': 0.995
            }
        },
        "standard": {  # Balanced training
            'num_episodes': 1000,
            'save_freq': 100,
            'eval_freq': 50
        },
        "deep": {  # Extensive training
            'num_episodes': 2500,
            'save_freq': 250,
            'eval_freq': 50,
            'agent_config': {
                # Optimized hyperparameters from Phase 4 Part 2
                'buffer_size': 100000,
                'epsilon_decay': 0.9995,
                'batch_size': 64,
                'learning_rate': 0.0005,
                'gamma': 0.99
            },
            'eval_tile_density': 0.8,
            'curriculum': {
                'enabled': True,
                'stages': [
                    {'start_episode': 1, 'tile_density': 0.5},
                    {'start_episode': 801, 'tile_density': 0.8},
                    {'start_episode': 1601, 'tile_density': 1.1}
                ],
                'adaptive': True,
                'advance_win_rate': 0.9,
                'window': 50
            }
        }
    }
    
    return configs.get(config_type, configs["standard"])

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train DQN Agent for 2D Car Racing')
    parser.add_argument('--config', type=str, default='standard',
                      choices=['fast', 'standard', 'deep'],
                      help='Training configuration preset')
    parser.add_argument('--episodes', type=int, help='Number of training episodes')
    parser.add_argument('--render', action='store_true', help='Render training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='models', help='Model save directory')
    
    args = parser.parse_args()
    
    # Create configuration
    config = create_training_config(args.config)
    
    # Override with command line arguments
    if args.episodes:
        config['num_episodes'] = args.episodes
    if args.render:
        config['render_training'] = True
    if args.seed:
        config['seed'] = args.seed
    if args.save_dir:
        config['save_dir'] = args.save_dir
    
    print("=== DQN 2D Car Racing Training ===")
    print(f"Configuration: {args.config}")
    print(f"Episodes: {config['num_episodes']}")
    
    # Initialize and run training
    trainer = TrainingManager(config)
    
    try:
        trainer.setup()
        final_results = trainer.train()
        
        print("\n=== Training Summary ===")
        print(f"Final Win Rate: {final_results['win_rate']:.1%}")
        print(f"Final Average Reward: {final_results['avg_reward']:.2f}")
        print(f"Time Improvement vs Baseline: {final_results['time_improvement']:+.2f}s")
        
        if final_results['win_rate'] > 0.6:
            print("🏆 SUCCESS: DQN agent achieved >60% win rate!")
        elif final_results['time_improvement'] > 0:
            print("📈 PROGRESS: DQN agent shows improvement over baseline")
        else:
            print("🔄 Continue training or adjust hyperparameters")
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()
