#!/usr/bin/env python3
"""
Phase 4 Part 2: Hyperparameter Optimization
Systematic optimization of DQN hyperparameters for racing performance
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import subprocess
import sys

class HyperparameterOptimizer:
    """Systematic hyperparameter optimization for DQN racing agent"""
    
    def __init__(self, base_episodes: int = 500):
        self.base_episodes = base_episodes
        self.results = {}
        self.optimization_log = []
        
        # Create optimization directories
        self.opt_dir = "hyperopt_results"
        self.models_dir = f"{self.opt_dir}/models"
        self.logs_dir = f"{self.opt_dir}/logs"
        
        for dir_path in [self.opt_dir, self.models_dir, self.logs_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def run_training_experiment(self, config: Dict, experiment_name: str) -> Dict:
        """Run a training experiment with specific configuration"""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {experiment_name}")
        print(f"{'='*60}")
        print(f"Configuration: {config}")
        
        # Create custom config file
        config_path = f"{self.opt_dir}/config_{experiment_name}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run training with custom configuration
        start_time = time.time()
        
        try:
            # Build command
            cmd = [
                sys.executable, "train_dqn.py",
                "--episodes", str(config.get('episodes', self.base_episodes)),
                "--save-dir", self.models_dir
            ]
            
            # Add custom config parameters via environment variables
            env = os.environ.copy()
            env['DQN_LEARNING_RATE'] = str(config.get('learning_rate', 0.0005))
            env['DQN_EPSILON_DECAY'] = str(config.get('epsilon_decay', 0.9995))
            env['DQN_BUFFER_SIZE'] = str(config.get('buffer_size', 100000))
            env['DQN_BATCH_SIZE'] = str(config.get('batch_size', 64))
            env['DQN_HIDDEN_LAYERS'] = json.dumps(config.get('hidden_layers', [128, 128, 64]))
            
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  timeout=1800, env=env)  # 30 min timeout
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                # Parse training output for performance metrics
                output_lines = result.stdout.split('\\n')
                final_stats = self._parse_training_output(output_lines)
                
                # Run evaluation
                eval_results = self._evaluate_model(f"{self.models_dir}/dqn_racing_final.pth")
                
                experiment_result = {
                    'config': config,
                    'training_time': training_time,
                    'training_stats': final_stats,
                    'evaluation': eval_results,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"✅ EXPERIMENT COMPLETED")
                print(f"   Training Time: {training_time:.1f}s")
                print(f"   Final Win Rate: {eval_results.get('win_rate', 0):.1f}%")
                print(f"   Avg Reward: {eval_results.get('avg_reward', 0):.1f}")
                
            else:
                print(f"❌ EXPERIMENT FAILED: {result.stderr}")
                experiment_result = {
                    'config': config,
                    'success': False,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
        
        except subprocess.TimeoutExpired:
            print(f"⏰ EXPERIMENT TIMEOUT")
            experiment_result = {
                'config': config,
                'success': False,
                'error': 'Timeout after 30 minutes',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"💥 EXPERIMENT ERROR: {e}")
            experiment_result = {
                'config': config,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        # Save experiment result
        result_path = f"{self.logs_dir}/{experiment_name}.json"
        with open(result_path, 'w') as f:
            json.dump(experiment_result, f, indent=2)
        
        self.results[experiment_name] = experiment_result
        self.optimization_log.append(experiment_result)
        
        return experiment_result
    
    def _parse_training_output(self, lines: List[str]) -> Dict:
        """Parse training output for key metrics"""
        stats = {'final_win_rate': 0, 'final_reward': 0, 'episodes_completed': 0}
        
        for line in lines:
            if 'Final Win Rate:' in line:
                try:
                    stats['final_win_rate'] = float(line.split('Final Win Rate:')[1].split('%')[0])
                except:
                    pass
            elif 'Final Average Reward:' in line:
                try:
                    stats['final_reward'] = float(line.split('Final Average Reward:')[1].split()[0])
                except:
                    pass
            elif 'Training completed' in line:
                stats['training_completed'] = True
        
        return stats
    
    def _evaluate_model(self, model_path: str, num_races: int = 10) -> Dict:
        """Evaluate trained model performance"""
        try:
            cmd = [sys.executable, "demo_trained.py", "--headless", 
                   "--model", model_path, "--races", str(num_races)]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse evaluation output
                lines = result.stdout.split('\\n')
                win_rate = 0
                avg_reward = 0
                
                for line in lines:
                    if 'Win rate:' in line:
                        try:
                            win_rate = float(line.split('Win rate:')[1].split('%')[0])
                        except:
                            pass
                    elif 'Average reward:' in line:
                        try:
                            avg_reward = float(line.split('Average reward:')[1].strip())
                        except:
                            pass
                
                return {
                    'win_rate': win_rate,
                    'avg_reward': avg_reward,
                    'num_races': num_races,
                    'success': True
                }
            else:
                return {'success': False, 'error': result.stderr}
                
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def optimize_learning_rate(self) -> Dict:
        """2.1 Learning Rate Optimization"""
        print(f"\n🔬 PHASE 2.1: LEARNING RATE OPTIMIZATION")
        print(f"Testing learning rates: [0.0001, 0.0005, 0.001, 0.002]")
        
        learning_rates = [0.0001, 0.0005, 0.001, 0.002]
        lr_results = {}
        
        for lr in learning_rates:
            config = {
                'learning_rate': lr,
                'episodes': self.base_episodes,
                'epsilon_decay': 0.9995,  # Keep other params constant
                'buffer_size': 100000,
                'batch_size': 64,
                'hidden_layers': [128, 128, 64]
            }
            
            experiment_name = f"lr_{lr}"
            result = self.run_training_experiment(config, experiment_name)
            lr_results[lr] = result
        
        # Find best learning rate
        best_lr = self._find_best_config(lr_results, 'learning_rate')
        print(f"\n🏆 BEST LEARNING RATE: {best_lr}")
        
        return lr_results
    
    def optimize_network_architecture(self, best_lr: float) -> Dict:
        """2.2 Network Architecture Experiments"""
        print(f"\n🧠 PHASE 2.2: NETWORK ARCHITECTURE OPTIMIZATION")
        
        architectures = {
            'current': [128, 128, 64],
            'smaller': [64, 64, 32],
            'larger': [256, 128, 64],
            'simpler': [128, 64]
        }
        
        arch_results = {}
        
        for name, layers in architectures.items():
            config = {
                'learning_rate': best_lr,
                'episodes': self.base_episodes,
                'epsilon_decay': 0.9995,
                'buffer_size': 100000,
                'batch_size': 64,
                'hidden_layers': layers
            }
            
            experiment_name = f"arch_{name}"
            result = self.run_training_experiment(config, experiment_name)
            arch_results[name] = result
        
        best_arch = self._find_best_config(arch_results, 'architecture')
        print(f"\n🏆 BEST ARCHITECTURE: {best_arch}")
        
        return arch_results
    
    def _find_best_config(self, results: Dict, param_type: str) -> str:
        """Find best configuration based on evaluation results"""
        best_config = None
        best_score = 0
        
        for name, result in results.items():
            if result['success'] and 'evaluation' in result:
                eval_data = result['evaluation']
                if eval_data['success']:
                    # Score based on win rate + reward
                    score = eval_data['win_rate'] + eval_data['avg_reward'] / 1000
                    if score > best_score:
                        best_score = score
                        best_config = name
        
        return best_config or list(results.keys())[0]
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        print(f"\n📊 GENERATING OPTIMIZATION REPORT...")
        
        report = {
            'phase': 'Phase 4 Part 2 - Hyperparameter Optimization',
            'timestamp': datetime.now().isoformat(),
            'base_episodes': self.base_episodes,
            'experiments': len(self.results),
            'results': self.results,
            'summary': self._create_summary()
        }
        
        report_path = f"{self.opt_dir}/PHASE4_PART2_OPTIMIZATION_REPORT.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Create markdown report
        self._create_markdown_report()
        
        print(f"✅ Optimization report saved to: {report_path}")
    
    def _create_summary(self) -> Dict:
        """Create optimization summary"""
        successful_experiments = [r for r in self.results.values() if r['success']]
        
        if not successful_experiments:
            return {'status': 'No successful experiments'}
        
        best_performance = max(
            successful_experiments,
            key=lambda x: x['evaluation']['win_rate'] + x['evaluation']['avg_reward']/1000
            if x.get('evaluation', {}).get('success') else 0
        )
        
        return {
            'total_experiments': len(self.results),
            'successful_experiments': len(successful_experiments),
            'best_performance': {
                'config': best_performance['config'],
                'win_rate': best_performance['evaluation']['win_rate'],
                'avg_reward': best_performance['evaluation']['avg_reward']
            }
        }
    
    def _create_markdown_report(self):
        """Create detailed markdown optimization report"""
        report_content = f"""# Phase 4 Part 2: Hyperparameter Optimization Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Base Episodes per Experiment:** {self.base_episodes}  
**Total Experiments:** {len(self.results)}

## Optimization Results Summary

"""
        
        # Add detailed results for each experiment
        for name, result in self.results.items():
            status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
            report_content += f"### {name.upper()}\n"
            report_content += f"**Status:** {status}\\n"
            
            if result['success'] and 'evaluation' in result:
                eval_data = result['evaluation']
                report_content += f"**Win Rate:** {eval_data.get('win_rate', 0):.1f}%\\n"
                report_content += f"**Average Reward:** {eval_data.get('avg_reward', 0):.1f}\\n"
                report_content += f"**Training Time:** {result.get('training_time', 0):.1f}s\\n"
            
            report_content += f"**Configuration:** `{result['config']}`\\n\\n"
        
        # Save markdown report
        with open(f"{self.opt_dir}/PHASE4_PART2_OPTIMIZATION_REPORT.md", 'w') as f:
            f.write(report_content)

def main():
    """Run hyperparameter optimization"""
    print("🚀 PHASE 4 PART 2: HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    optimizer = HyperparameterOptimizer(base_episodes=500)
    
    try:
        # 2.1 Learning Rate Optimization
        lr_results = optimizer.optimize_learning_rate()
        best_lr = optimizer._find_best_config(lr_results, 'learning_rate')
        best_lr_value = lr_results[best_lr]['config']['learning_rate']
        
        # 2.2 Network Architecture Optimization  
        arch_results = optimizer.optimize_network_architecture(best_lr_value)
        
        # Generate final report
        optimizer.generate_optimization_report()
        
        print(f"\n🎉 HYPERPARAMETER OPTIMIZATION COMPLETED!")
        print(f"📊 Check {optimizer.opt_dir}/ for detailed results")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Optimization interrupted by user")
        optimizer.generate_optimization_report()
    except Exception as e:
        print(f"\n💥 Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()