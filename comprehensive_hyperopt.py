#!/usr/bin/env python3
"""
Phase 4 Part 2: Comprehensive Hyperparameter Optimization
Complete implementation of all optimization categories from Phase 4 Part 2
"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__)))

from train_dqn import TrainingManager, create_training_config

@dataclass
class OptimizationResult:
    """Result from a single hyperparameter configuration experiment"""
    config_id: str
    category: str
    hyperparameters: Dict[str, Any]
    final_results: Dict[str, float] = None
    training_duration: float = 0.0
    success: bool = False
    error_message: str = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()

class ComprehensiveHyperparameterOptimizer:
    """Complete hyperparameter optimization system covering all Phase 4 Part 2 categories"""
    
    def __init__(self, base_episodes: int = 500, quick_mode: bool = False):
        """Initialize comprehensive optimizer
        
        Args:
            base_episodes: Number of episodes for each experiment
            quick_mode: Use fewer episodes for faster testing
        """
        self.base_episodes = 300 if quick_mode else base_episodes
        self.quick_mode = quick_mode
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup directories
        self.results_dir = Path("hyperopt_results")
        self.models_dir = self.results_dir / "models"
        self.logs_dir = self.results_dir / "logs"
        self.plots_dir = self.results_dir / "plots"
        
        for directory in [self.results_dir, self.models_dir, self.logs_dir, self.plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Optimization results storage
        self.all_results: List[OptimizationResult] = []
        self.best_configs: Dict[str, OptimizationResult] = {}
        
        print(f"🚀 Comprehensive Hyperparameter Optimization Session: {self.session_id}")
        print(f"📁 Results Directory: {self.results_dir}")
        print(f"⚡ Quick Mode: {'Yes' if quick_mode else 'No'}")
        print(f"🏃 Episodes per experiment: {self.base_episodes}")
    
    def create_base_config(self) -> Dict:
        """Create base training configuration"""
        config = create_training_config("standard")
        config['num_episodes'] = self.base_episodes
        config['plot_training'] = False  # Disable plots during optimization
        config['render_training'] = False  # No rendering
        
        # Always ensure agent_config exists with complete default values
        config['agent_config'] = {
            'learning_rate': 0.0005,
            'gamma': 0.99,
            'epsilon': 1.0,
            'epsilon_min': 0.05,
            'epsilon_decay': 0.9995,
            'buffer_size': 100000,
            'batch_size': 64,
            'update_freq': 4,
            'target_update_freq': 1000,
            'network_type': 'double',
            'buffer_type': 'standard',
            'hidden_layers': [128, 128, 64]
        }
        
        return config
    
    def run_single_experiment(self, config: Dict, config_id: str, category: str, description: str) -> OptimizationResult:
        """Run a single hyperparameter experiment with detailed tracking"""
        print(f"   🔄 Running: {description}")
        start_time = time.time()
        
        # Setup unique directories
        experiment_dir = self.models_dir / config_id
        experiment_dir.mkdir(exist_ok=True)
        
        config['save_dir'] = str(experiment_dir)
        config['log_dir'] = str(self.logs_dir)
        
        try:
            # Run training
            trainer = TrainingManager(config)
            trainer.setup()
            final_results = trainer.train()
            trainer.cleanup()
            
            training_duration = time.time() - start_time
            
            # Create result
            result = OptimizationResult(
                config_id=config_id,
                category=category,
                hyperparameters=config['agent_config'].copy(),
                final_results=final_results,
                training_duration=training_duration,
                success=True
            )
            
            print(f"   ✅ Success: Win Rate: {final_results['win_rate']:.1%}, "
                  f"Time Improvement: {final_results['time_improvement']:+.2f}s, "
                  f"Duration: {training_duration:.0f}s")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
            result = OptimizationResult(
                config_id=config_id,
                category=category,
                hyperparameters=config['agent_config'].copy(),
                training_duration=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
        
        # Save individual result
        self.save_individual_result(result)
        return result
    
    def save_individual_result(self, result: OptimizationResult):
        """Save individual experiment result"""
        result_file = self.logs_dir / f"{result.config_id}.json"
        with open(result_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)
    
    def find_best_result(self, results: List[OptimizationResult]) -> OptimizationResult:
        """Find best result based on performance metrics"""
        successful_results = [r for r in results if r.success and r.final_results]
        
        if not successful_results:
            return None
        
        def score_function(result):
            final_results = result.final_results
            win_rate = final_results['win_rate']
            time_improvement = final_results['time_improvement']
            # Weighted score: win rate most important, time improvement secondary
            return win_rate * 1000 + max(0, time_improvement) * 10
        
        return max(successful_results, key=score_function)
    
    # =====================================
    # 2.1 LEARNING RATE OPTIMIZATION
    # =====================================
    
    def optimize_learning_rate(self) -> List[OptimizationResult]:
        """2.1 Learning Rate Grid Search: [0.0001, 0.0005, 0.001, 0.002]"""
        print("\n" + "="*80)
        print("🧮 2.1 LEARNING RATE OPTIMIZATION")
        print("="*80)
        print("Testing learning rates: [0.0001, 0.0005, 0.001, 0.002]")
        print(f"Episodes per configuration: {self.base_episodes}")
        
        learning_rates = [0.0001, 0.0005, 0.001, 0.002]
        results = []
        
        for i, lr in enumerate(learning_rates):
            print(f"\n📊 Experiment {i+1}/4: Learning Rate = {lr}")
            
            config = self.create_base_config()
            config['agent_config']['learning_rate'] = lr
            
            config_id = f"lr_{lr}_{self.session_id}"
            result = self.run_single_experiment(config, config_id, "learning_rate", f"Learning Rate {lr}")
            results.append(result)
            self.all_results.append(result)
        
        # Find best learning rate
        best_result = self.find_best_result(results)
        if best_result:
            self.best_configs['learning_rate'] = best_result
            print(f"\n🏆 BEST LEARNING RATE: {best_result.hyperparameters['learning_rate']}")
            print(f"   Win Rate: {best_result.final_results['win_rate']:.1%}")
            print(f"   Time Improvement: {best_result.final_results['time_improvement']:+.2f}s")
        
        return results
    
    # =====================================
    # 2.2 NETWORK ARCHITECTURE OPTIMIZATION
    # =====================================
    
    def optimize_network_architecture(self, best_lr: float = None) -> List[OptimizationResult]:
        """2.2 Network Architecture Experiments: Test different hidden layer configurations"""
        print("\n" + "="*80)
        print("🧠 2.2 NETWORK ARCHITECTURE OPTIMIZATION")
        print("="*80)
        
        if best_lr:
            print(f"Using best learning rate: {best_lr}")
        else:
            best_lr = 0.0005  # Default
            print(f"Using default learning rate: {best_lr}")
        
        architectures = {
            'smaller_fast': [64, 64, 32],      # Smaller, faster
            'current_baseline': [128, 128, 64], # Current baseline
            'larger_capacity': [256, 128, 64],  # Larger, more capacity
            'simpler': [128, 64]               # Simpler architecture
        }
        
        results = []
        
        for i, (arch_name, layers) in enumerate(architectures.items()):
            print(f"\n🏗️  Experiment {i+1}/4: Architecture '{arch_name}' = {layers}")
            
            config = self.create_base_config()
            config['agent_config']['learning_rate'] = best_lr
            config['agent_config']['hidden_layers'] = layers
            
            config_id = f"arch_{arch_name}_{self.session_id}"
            result = self.run_single_experiment(config, config_id, "architecture", f"Architecture {arch_name}")
            results.append(result)
            self.all_results.append(result)
        
        # Find best architecture
        best_result = self.find_best_result(results)
        if best_result:
            self.best_configs['architecture'] = best_result
            print(f"\n🏆 BEST ARCHITECTURE: {best_result.hyperparameters['hidden_layers']}")
            print(f"   Win Rate: {best_result.final_results['win_rate']:.1%}")
            print(f"   Time Improvement: {best_result.final_results['time_improvement']:+.2f}s")
        
        return results
    
    # =====================================
    # 2.3 EXPLORATION STRATEGY OPTIMIZATION
    # =====================================
    
    def optimize_exploration_strategy(self, best_lr: float = None, best_arch: List[int] = None) -> List[OptimizationResult]:
        """2.3 Exploration Strategy Tuning: Test epsilon decay rates and min values"""
        print("\n" + "="*80)
        print("🎯 2.3 EXPLORATION STRATEGY OPTIMIZATION")
        print("="*80)
        
        if best_lr:
            print(f"Using best learning rate: {best_lr}")
        else:
            best_lr = 0.0005
        
        if best_arch:
            print(f"Using best architecture: {best_arch}")
        else:
            best_arch = [128, 128, 64]
        
        exploration_configs = [
            {'decay': 0.999, 'min': 0.01, 'name': 'conservative'},  # Conservative exploration
            {'decay': 0.9995, 'min': 0.05, 'name': 'balanced'},    # Balanced exploration (current)
            {'decay': 0.995, 'min': 0.1, 'name': 'aggressive'}     # Aggressive exploitation
        ]
        
        results = []
        
        for i, exp_config in enumerate(exploration_configs):
            print(f"\n🎲 Experiment {i+1}/3: {exp_config['name'].title()} Exploration")
            print(f"   Epsilon Decay: {exp_config['decay']}, Min: {exp_config['min']}")
            
            config = self.create_base_config()
            config['agent_config']['learning_rate'] = best_lr
            config['agent_config']['hidden_layers'] = best_arch
            config['agent_config']['epsilon_decay'] = exp_config['decay']
            config['agent_config']['epsilon_min'] = exp_config['min']
            
            config_id = f"explore_{exp_config['name']}_{self.session_id}"
            result = self.run_single_experiment(config, config_id, "exploration", f"Exploration {exp_config['name']}")
            results.append(result)
            self.all_results.append(result)
        
        # Find best exploration strategy
        best_result = self.find_best_result(results)
        if best_result:
            self.best_configs['exploration'] = best_result
            print(f"\n🏆 BEST EXPLORATION STRATEGY:")
            print(f"   Epsilon Decay: {best_result.hyperparameters['epsilon_decay']}")
            print(f"   Epsilon Min: {best_result.hyperparameters['epsilon_min']}")
            print(f"   Win Rate: {best_result.final_results['win_rate']:.1%}")
        
        return results
    
    # =====================================
    # 2.4 EXPERIENCE REPLAY OPTIMIZATION
    # =====================================
    
    def optimize_experience_replay(self, best_params: Dict = None) -> List[OptimizationResult]:
        """2.4 Experience Replay Optimization: Test buffer sizes and batch sizes"""
        print("\n" + "="*80)
        print("💾 2.4 EXPERIENCE REPLAY OPTIMIZATION")
        print("="*80)
        
        replay_configs = [
            {'buffer_size': 50000, 'batch_size': 32, 'name': 'small'},
            {'buffer_size': 100000, 'batch_size': 64, 'name': 'medium'},
            {'buffer_size': 200000, 'batch_size': 128, 'name': 'large'}
        ]
        
        results = []
        
        for i, replay_config in enumerate(replay_configs):
            print(f"\n🗄️  Experiment {i+1}/3: {replay_config['name'].title()} Replay Buffer")
            print(f"   Buffer Size: {replay_config['buffer_size']:,}, Batch Size: {replay_config['batch_size']}")
            
            config = self.create_base_config()
            
            # Apply best parameters from previous optimizations
            if best_params:
                for key, value in best_params.items():
                    config['agent_config'][key] = value
            
            config['agent_config']['buffer_size'] = replay_config['buffer_size']
            config['agent_config']['batch_size'] = replay_config['batch_size']
            
            config_id = f"replay_{replay_config['name']}_{self.session_id}"
            result = self.run_single_experiment(config, config_id, "replay", f"Replay {replay_config['name']}")
            results.append(result)
            self.all_results.append(result)
        
        # Find best replay configuration
        best_result = self.find_best_result(results)
        if best_result:
            self.best_configs['replay'] = best_result
            print(f"\n🏆 BEST REPLAY CONFIGURATION:")
            print(f"   Buffer Size: {best_result.hyperparameters['buffer_size']:,}")
            print(f"   Batch Size: {best_result.hyperparameters['batch_size']}")
            print(f"   Win Rate: {best_result.final_results['win_rate']:.1%}")
        
        return results
    
    # =====================================
    # 2.5 REWARD FUNCTION OPTIMIZATION
    # =====================================
    
    def optimize_reward_function(self, best_params: Dict = None) -> List[OptimizationResult]:
        """2.5 Reward Function Tuning: Test gamma values as reward-related parameter"""
        print("\n" + "="*80)
        print("🎁 2.5 REWARD FUNCTION OPTIMIZATION")
        print("="*80)
        print("Testing gamma values (discount factor) as reward-related parameter")
        
        gamma_configs = [
            {'gamma': 0.95, 'name': 'short_term'},
            {'gamma': 0.99, 'name': 'balanced'},
            {'gamma': 0.995, 'name': 'long_term'}
        ]
        
        results = []
        
        for i, gamma_config in enumerate(gamma_configs):
            print(f"\n🎯 Experiment {i+1}/3: {gamma_config['name'].title()} Rewards (gamma={gamma_config['gamma']})")
            
            config = self.create_base_config()
            
            # Apply best parameters from previous optimizations
            if best_params:
                for key, value in best_params.items():
                    config['agent_config'][key] = value
            
            config['agent_config']['gamma'] = gamma_config['gamma']
            
            config_id = f"gamma_{gamma_config['name']}_{self.session_id}"
            result = self.run_single_experiment(config, config_id, "gamma", f"Gamma {gamma_config['name']}")
            results.append(result)
            self.all_results.append(result)
        
        # Find best gamma value
        best_result = self.find_best_result(results)
        if best_result:
            self.best_configs['gamma'] = best_result
            print(f"\n🏆 BEST GAMMA VALUE: {best_result.hyperparameters['gamma']}")
            print(f"   Win Rate: {best_result.final_results['win_rate']:.1%}")
        
        return results
    
    # =====================================
    # MAIN OPTIMIZATION PIPELINE
    # =====================================
    
    def run_comprehensive_optimization(self) -> Dict[str, Any]:
        """Run the complete Phase 4 Part 2 hyperparameter optimization pipeline"""
        print("\n" + "🎭" * 40)
        print("PHASE 4 PART 2: COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
        print("🎭" * 40)
        print(f"Session: {self.session_id}")
        print(f"Episodes per Experiment: {self.base_episodes}")
        print(f"Expected Total Duration: {self.estimate_total_duration():.0f} minutes")
        
        total_start_time = time.time()
        
        try:
            # 2.1 Learning Rate Optimization
            lr_results = self.optimize_learning_rate()
            best_lr = self.best_configs.get('learning_rate')
            best_lr_value = best_lr.hyperparameters['learning_rate'] if best_lr else 0.0005
            
            # 2.2 Network Architecture Optimization
            arch_results = self.optimize_network_architecture(best_lr_value)
            best_arch = self.best_configs.get('architecture')
            best_arch_value = best_arch.hyperparameters['hidden_layers'] if best_arch else [128, 128, 64]
            
            # 2.3 Exploration Strategy Optimization
            explore_results = self.optimize_exploration_strategy(best_lr_value, best_arch_value)
            
            # Collect best parameters so far
            best_params = {
                'learning_rate': best_lr_value,
                'hidden_layers': best_arch_value
            }
            
            if 'exploration' in self.best_configs:
                best_exp = self.best_configs['exploration']
                best_params['epsilon_decay'] = best_exp.hyperparameters['epsilon_decay']
                best_params['epsilon_min'] = best_exp.hyperparameters['epsilon_min']
            
            # 2.4 Experience Replay Optimization
            replay_results = self.optimize_experience_replay(best_params)
            
            # Update best parameters
            if 'replay' in self.best_configs:
                best_replay = self.best_configs['replay']
                best_params['buffer_size'] = best_replay.hyperparameters['buffer_size']
                best_params['batch_size'] = best_replay.hyperparameters['batch_size']
            
            # 2.5 Reward Function Optimization
            reward_results = self.optimize_reward_function(best_params)
            
            # Generate final report
            total_duration = time.time() - total_start_time
            final_report = self.generate_comprehensive_report(total_duration)
            
            print(f"\n" + "🎉" * 40)
            print("COMPREHENSIVE HYPERPARAMETER OPTIMIZATION COMPLETED!")
            print("🎉" * 40)
            print(f"Total Duration: {total_duration:.0f} seconds ({total_duration/60:.1f} minutes)")
            print(f"Total Experiments: {len(self.all_results)}")
            print(f"Successful Experiments: {len([r for r in self.all_results if r.success])}")
            print(f"Results saved to: {self.results_dir}")
            
            return final_report
            
        except KeyboardInterrupt:
            print("\n⏹️  Optimization interrupted by user")
            return self.generate_interrupted_report()
        except Exception as e:
            print(f"\n💥 Optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
    
    def estimate_total_duration(self) -> float:
        """Estimate total optimization duration in minutes"""
        # Rough estimates based on episode count
        minutes_per_episode = 0.1 if self.quick_mode else 0.15
        total_episodes = self.base_episodes * 17  # Total number of experiments
        return total_episodes * minutes_per_episode
    
    def generate_comprehensive_report(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive optimization report with all results"""
        print("\n📊 Generating comprehensive optimization report...")
        
        successful_results = [r for r in self.all_results if r.success]
        overall_best = self.find_best_result(successful_results) if successful_results else None
        
        # Create comprehensive report
        report = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 4 Part 2 - Comprehensive Hyperparameter Optimization',
            'total_duration_minutes': total_duration / 60,
            'base_episodes_per_experiment': self.base_episodes,
            'quick_mode': self.quick_mode,
            'total_experiments': len(self.all_results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len([r for r in self.all_results if not r.success]),
            'best_configs_by_category': {},
            'overall_best_config': None,
            'optimization_summary': self.create_optimization_summary(),
            'all_results': [asdict(r) for r in self.all_results]
        }
        
        # Add best configs by category
        for category, result in self.best_configs.items():
            report['best_configs_by_category'][category] = {
                'config_id': result.config_id,
                'hyperparameters': result.hyperparameters,
                'performance': result.final_results,
                'training_duration': result.training_duration
            }
        
        # Add overall best config
        if overall_best:
            report['overall_best_config'] = {
                'config_id': overall_best.config_id,
                'hyperparameters': overall_best.hyperparameters,
                'performance': overall_best.final_results,
                'training_duration': overall_best.training_duration
            }
            
            print(f"\n🏆 OVERALL BEST CONFIGURATION:")
            print(f"   Config ID: {overall_best.config_id}")
            print(f"   Win Rate: {overall_best.final_results['win_rate']:.1%}")
            print(f"   Time Improvement: {overall_best.final_results['time_improvement']:+.2f}s")
            print(f"   Learning Rate: {overall_best.hyperparameters['learning_rate']}")
            print(f"   Architecture: {overall_best.hyperparameters['hidden_layers']}")
        
        # Save report
        report_file = self.results_dir / f"COMPREHENSIVE_HYPEROPT_REPORT_{self.session_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report and plots
        self.generate_markdown_report(report)
        self.generate_optimization_plots()
        
        print(f"✅ Comprehensive optimization report saved to: {report_file}")
        
        return report
    
    def create_optimization_summary(self) -> Dict[str, Any]:
        """Create optimization summary statistics"""
        successful_results = [r for r in self.all_results if r.success]
        
        if not successful_results:
            return {'status': 'No successful experiments'}
        
        win_rates = [r.final_results['win_rate'] for r in successful_results]
        time_improvements = [r.final_results['time_improvement'] for r in successful_results]
        durations = [r.training_duration for r in successful_results]
        
        return {
            'win_rate_stats': {
                'mean': np.mean(win_rates),
                'std': np.std(win_rates),
                'min': np.min(win_rates),
                'max': np.max(win_rates)
            },
            'time_improvement_stats': {
                'mean': np.mean(time_improvements),
                'std': np.std(time_improvements),
                'min': np.min(time_improvements),
                'max': np.max(time_improvements)
            },
            'training_duration_stats': {
                'mean': np.mean(durations),
                'total': np.sum(durations)
            },
            'categories_tested': list(self.best_configs.keys())
        }
    
    def generate_markdown_report(self, report: Dict[str, Any]):
        """Generate detailed markdown report"""
        markdown_content = f"""# Phase 4 Part 2: Comprehensive Hyperparameter Optimization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Session ID:** {self.session_id}  
**Episodes per Experiment:** {self.base_episodes}  
**Total Duration:** {report['total_duration_minutes']:.1f} minutes  
**Quick Mode:** {'Yes' if self.quick_mode else 'No'}

## Executive Summary

This comprehensive hyperparameter optimization tested all five categories from Phase 4 Part 2:
1. **Learning Rate Optimization** - Testing [0.0001, 0.0005, 0.001, 0.002]
2. **Network Architecture** - Testing different hidden layer configurations
3. **Exploration Strategy** - Testing epsilon decay rates and minimum values
4. **Experience Replay** - Testing buffer sizes and batch sizes
5. **Reward Function** - Testing gamma values (discount factor)

## Optimization Results

- **Total Experiments:** {report['total_experiments']}
- **Successful:** {report['successful_experiments']}
- **Failed:** {report['failed_experiments']}
- **Success Rate:** {report['successful_experiments']/report['total_experiments']:.1%}

"""
        
        # Overall best configuration
        if report['overall_best_config']:
            best = report['overall_best_config']
            markdown_content += f"""## 🏆 Overall Best Configuration

- **Config ID:** `{best['config_id']}`
- **Win Rate:** {best['performance']['win_rate']:.1%}
- **Time Improvement:** {best['performance']['time_improvement']:+.2f}s
- **Average Reward:** {best['performance']['avg_reward']:.2f}
- **Training Duration:** {best['training_duration']:.0f}s

**Optimal Hyperparameters:**
```json
{json.dumps(best['hyperparameters'], indent=2)}
```

"""
        
        # Best configurations by category
        markdown_content += "## Best Configurations by Category\n\n"
        
        for category, config_info in report['best_configs_by_category'].items():
            markdown_content += f"""### {category.upper().replace('_', ' ')}

- **Config ID:** `{config_info['config_id']}`
- **Win Rate:** {config_info['performance']['win_rate']:.1%}
- **Time Improvement:** {config_info['performance']['time_improvement']:+.2f}s
- **Training Duration:** {config_info['training_duration']:.0f}s

**Key Parameters:**
"""
            # Show relevant parameters for each category
            if category == 'learning_rate':
                markdown_content += f"- Learning Rate: {config_info['hyperparameters']['learning_rate']}\n"
            elif category == 'architecture':
                markdown_content += f"- Hidden Layers: {config_info['hyperparameters']['hidden_layers']}\n"
            elif category == 'exploration':
                markdown_content += f"- Epsilon Decay: {config_info['hyperparameters']['epsilon_decay']}\n"
                markdown_content += f"- Epsilon Min: {config_info['hyperparameters']['epsilon_min']}\n"
            elif category == 'replay':
                markdown_content += f"- Buffer Size: {config_info['hyperparameters']['buffer_size']:,}\n"
                markdown_content += f"- Batch Size: {config_info['hyperparameters']['batch_size']}\n"
            elif category == 'gamma':
                markdown_content += f"- Gamma: {config_info['hyperparameters']['gamma']}\n"
            
            markdown_content += "\n"
        
        # Performance statistics
        if 'optimization_summary' in report:
            stats = report['optimization_summary']
            if 'win_rate_stats' in stats:
                markdown_content += f"""## Performance Statistics

### Win Rate Analysis
- **Mean Win Rate:** {stats['win_rate_stats']['mean']:.1%}
- **Best Win Rate:** {stats['win_rate_stats']['max']:.1%}
- **Standard Deviation:** {stats['win_rate_stats']['std']:.3f}

### Time Improvement Analysis
- **Mean Time Improvement:** {stats['time_improvement_stats']['mean']:+.2f}s
- **Best Time Improvement:** {stats['time_improvement_stats']['max']:+.2f}s
- **Standard Deviation:** {stats['time_improvement_stats']['std']:.3f}

### Training Efficiency
- **Average Training Duration:** {stats['training_duration_stats']['mean']:.0f}s
- **Total Training Time:** {stats['training_duration_stats']['total']/3600:.2f} hours

"""
        
        markdown_content += f"""## Conclusions and Recommendations

Based on the comprehensive hyperparameter optimization:

1. **Best Learning Rate:** Use the optimal learning rate identified for stable training
2. **Network Architecture:** The best-performing architecture balances model capacity and training efficiency
3. **Exploration Strategy:** Optimal epsilon parameters balance exploration vs exploitation
4. **Experience Replay:** Optimal buffer and batch sizes for efficient learning
5. **Reward Function:** Gamma value affects long-term vs short-term reward focus

## Next Steps

1. **Phase 4 Part 3:** Use these optimized hyperparameters for extended training (2000+ episodes)
2. **Statistical Validation:** Run multiple seeds with best configuration
3. **Final Competition:** Tournament-style evaluation against baseline

---

*This comprehensive optimization provides a solid foundation for Phase 4 Part 3 advanced training.*
"""
        
        # Save markdown report
        markdown_file = self.results_dir / f"COMPREHENSIVE_HYPEROPT_REPORT_{self.session_id}.md"
        with open(markdown_file, 'w') as f:
            f.write(markdown_content)
        
        print(f"📝 Markdown report saved to: {markdown_file}")
    
    def generate_optimization_plots(self):
        """Generate comprehensive visualization plots"""
        successful_results = [r for r in self.all_results if r.success]
        
        if not successful_results:
            print("⚠️  No successful results to plot")
            return
        
        # Extract data for plotting
        categories = [r.category for r in successful_results]
        win_rates = [r.final_results['win_rate'] for r in successful_results]
        time_improvements = [r.final_results['time_improvement'] for r in successful_results]
        durations = [r.training_duration for r in successful_results]
        
        # Create comprehensive plots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Win Rate Distribution
        plt.subplot(2, 3, 1)
        plt.hist(win_rates, bins=min(10, len(win_rates)), alpha=0.7, color='green')
        plt.title('Win Rate Distribution')
        plt.xlabel('Win Rate')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # 2. Time Improvement Distribution
        plt.subplot(2, 3, 2)
        plt.hist(time_improvements, bins=min(10, len(time_improvements)), alpha=0.7, color='blue')
        plt.title('Time Improvement Distribution')
        plt.xlabel('Time Improvement (seconds)')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        
        # 3. Performance by Category
        plt.subplot(2, 3, 3)
        unique_categories = list(set(categories))
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_categories)))
        
        for i, category in enumerate(unique_categories):
            category_wr = [wr for cat, wr in zip(categories, win_rates) if cat == category]
            category_ti = [ti for cat, ti in zip(categories, time_improvements) if cat == category]
            plt.scatter(category_ti, category_wr, color=colors[i], label=category, alpha=0.7, s=60)
        
        plt.title('Performance by Category')
        plt.xlabel('Time Improvement (seconds)')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 4. Training Duration by Category
        plt.subplot(2, 3, 4)
        category_durations = {}
        for cat, dur in zip(categories, durations):
            if cat not in category_durations:
                category_durations[cat] = []
            category_durations[cat].append(dur)
        
        cat_names = list(category_durations.keys())
        avg_durations = [np.mean(category_durations[cat]) for cat in cat_names]
        
        plt.bar(cat_names, avg_durations, alpha=0.7, color='coral')
        plt.title('Average Training Duration by Category')
        plt.xlabel('Category')
        plt.ylabel('Duration (seconds)')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 5. Win Rate vs Time Improvement Detailed
        plt.subplot(2, 3, 5)
        plt.scatter(time_improvements, win_rates, c=[colors[unique_categories.index(cat)] for cat in categories], alpha=0.7, s=60)
        plt.title('Win Rate vs Time Improvement (All Results)')
        plt.xlabel('Time Improvement (seconds)')
        plt.ylabel('Win Rate')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        if len(time_improvements) > 1:
            z = np.polyfit(time_improvements, win_rates, 1)
            p = np.poly1d(z)
            plt.plot(sorted(time_improvements), p(sorted(time_improvements)), "r--", alpha=0.8)
        
        # 6. Best Configuration Summary
        plt.subplot(2, 3, 6)
        best_configs = list(self.best_configs.values())
        if best_configs:
            best_categories = [result.category for result in best_configs]
            best_win_rates = [result.final_results['win_rate'] for result in best_configs]
            
            bars = plt.bar(best_categories, best_win_rates, alpha=0.8, color='gold')
            plt.title('Best Win Rate by Category')
            plt.xlabel('Category')
            plt.ylabel('Win Rate')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, rate in zip(bars, best_win_rates):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.plots_dir / f"comprehensive_optimization_analysis_{self.session_id}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Comprehensive optimization plots saved to: {plot_file}")
    
    def generate_interrupted_report(self) -> Dict[str, Any]:
        """Generate report for interrupted optimization"""
        return {
            'session_id': self.session_id,
            'status': 'interrupted',
            'completed_experiments': len(self.all_results),
            'successful_experiments': len([r for r in self.all_results if r.success]),
            'best_configs_found': list(self.best_configs.keys()),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main function for comprehensive hyperparameter optimization"""
    parser = argparse.ArgumentParser(description='Comprehensive DQN Hyperparameter Optimization')
    parser.add_argument('--episodes', type=int, default=500,
                      help='Episodes per experiment (default: 500)')
    parser.add_argument('--quick', action='store_true',
                      help='Quick optimization with fewer episodes (300)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 PHASE 4 PART 2: COMPREHENSIVE HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("Complete implementation covering all optimization categories:")
    print("  2.1 Learning Rate Optimization")
    print("  2.2 Network Architecture Experiments")
    print("  2.3 Exploration Strategy Tuning")
    print("  2.4 Experience Replay Optimization")
    print("  2.5 Reward Function Tuning")
    print("=" * 80)
    
    # Initialize comprehensive optimizer
    optimizer = ComprehensiveHyperparameterOptimizer(
        base_episodes=args.episodes, 
        quick_mode=args.quick
    )
    
    try:
        final_report = optimizer.run_comprehensive_optimization()
        
        print(f"\n✅ COMPREHENSIVE OPTIMIZATION COMPLETED SUCCESSFULLY")
        print(f"📊 Results saved to: {optimizer.results_dir}")
        
        # Show final recommendations
        if 'overall_best_config' in final_report and final_report['overall_best_config']:
            best = final_report['overall_best_config']
            print(f"\n🏆 RECOMMENDED HYPERPARAMETERS FOR PHASE 4 PART 3:")
            print(f"   Learning Rate: {best['hyperparameters']['learning_rate']}")
            print(f"   Architecture: {best['hyperparameters']['hidden_layers']}")
            print(f"   Win Rate: {best['performance']['win_rate']:.1%}")
            print(f"   Time Improvement: {best['performance']['time_improvement']:+.2f}s")
            
            print(f"\n📋 Use these hyperparameters for extended training in Phase 4 Part 3")
        
    except KeyboardInterrupt:
        print("\n⏹️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n💥 Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()