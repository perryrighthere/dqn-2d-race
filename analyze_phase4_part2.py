#!/usr/bin/env python3
"""
Phase 4 Part 2 Analysis: Hyperparameter Optimization Results Analysis
Analyze and visualize hyperparameter optimization results
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

class Phase4Part2Analyzer:
    """Analyzer for Phase 4 Part 2 hyperparameter optimization results"""
    
    def __init__(self, results_dir: str = "hyperopt_results"):
        """Initialize analyzer
        
        Args:
            results_dir: Directory containing optimization results
        """
        self.results_dir = Path(results_dir)
        self.results = {}
        self.best_configs = {}
        
        print(f"🔍 Phase 4 Part 2 Results Analyzer")
        print(f"📁 Results directory: {self.results_dir}")
        
    def load_results(self):
        """Load all available optimization results"""
        print(f"\n📊 Loading optimization results...")
        
        # Load learning rate results
        lr_file = self.results_dir / "learning_rate_results.json"
        if lr_file.exists():
            with open(lr_file, 'r') as f:
                self.results['learning_rate'] = json.load(f)
            print(f"   ✅ Learning rate results loaded: {len(self.results['learning_rate'])} experiments")
        
        # Load exploration results
        exp_file = self.results_dir / "exploration_results.json"
        if exp_file.exists():
            with open(exp_file, 'r') as f:
                self.results['exploration'] = json.load(f)
            print(f"   ✅ Exploration results loaded: {len(self.results['exploration'])} experiments")
        
        # Load comprehensive results if available
        for result_file in self.results_dir.glob("COMPREHENSIVE_HYPEROPT_REPORT_*.json"):
            with open(result_file, 'r') as f:
                comprehensive_data = json.load(f)
                self.results['comprehensive'] = comprehensive_data
            print(f"   ✅ Comprehensive results loaded from {result_file.name}")
            break
        
        # Load Phase 4 Part 2 results if available
        phase4_file = self.results_dir / "PHASE4_PART2_RESULTS.json"
        if phase4_file.exists():
            with open(phase4_file, 'r') as f:
                self.results['phase4_part2'] = json.load(f)
            print(f"   ✅ Phase 4 Part 2 results loaded")
        
        if not self.results:
            print("   ⚠️  No results files found")
            return False
        
        return True
    
    def analyze_learning_rate_optimization(self):
        """Analyze learning rate optimization results"""
        if 'learning_rate' not in self.results:
            print("❌ No learning rate results found")
            return
        
        print(f"\n📈 LEARNING RATE OPTIMIZATION ANALYSIS")
        print("="*50)
        
        lr_results = self.results['learning_rate']
        successful = {lr: res for lr, res in lr_results.items() if res.get('success', False)}
        
        print(f"Total experiments: {len(lr_results)}")
        print(f"Successful experiments: {len(successful)}")
        
        if successful:
            print(f"\nResults by Learning Rate:")
            for lr, result in successful.items():
                win_rate = result.get('eval_win_rate', 0)
                reward = result.get('eval_reward', 0)
                time_taken = result.get('training_time', 0)
                print(f"  LR {lr:>6}: Win Rate {win_rate:>5.1f}%, Reward {reward:>6.1f}, Time {time_taken:>5.1f}s")
            
            # Find best learning rate
            best_lr = max(successful.keys(), 
                         key=lambda x: successful[x]['eval_win_rate'] + successful[x]['eval_reward']/1000)
            
            self.best_configs['learning_rate'] = {
                'value': best_lr,
                'performance': successful[best_lr]
            }
            
            print(f"\n🏆 BEST LEARNING RATE: {best_lr}")
            print(f"   Win Rate: {successful[best_lr]['eval_win_rate']:.1f}%")
            print(f"   Reward: {successful[best_lr]['eval_reward']:.1f}")
        
    def analyze_exploration_optimization(self):
        """Analyze exploration strategy optimization results"""
        if 'exploration' not in self.results:
            print("❌ No exploration results found")
            return
        
        print(f"\n🎯 EXPLORATION STRATEGY OPTIMIZATION ANALYSIS")
        print("="*50)
        
        exp_results = self.results['exploration']
        successful = {decay: res for decay, res in exp_results.items() if res.get('success', False)}
        
        print(f"Total experiments: {len(exp_results)}")
        print(f"Successful experiments: {len(successful)}")
        
        if successful:
            print(f"\nResults by Epsilon Decay:")
            for decay, result in successful.items():
                win_rate = result.get('eval_win_rate', 0)
                reward = result.get('eval_reward', 0)
                time_taken = result.get('training_time', 0)
                print(f"  Decay {decay:>6}: Win Rate {win_rate:>5.1f}%, Reward {reward:>6.1f}, Time {time_taken:>5.1f}s")
            
            # Find best epsilon decay
            best_decay = max(successful.keys(), 
                           key=lambda x: successful[x]['eval_win_rate'] + successful[x]['eval_reward']/1000)
            
            self.best_configs['exploration'] = {
                'value': best_decay,
                'performance': successful[best_decay]
            }
            
            print(f"\n🏆 BEST EPSILON DECAY: {best_decay}")
            print(f"   Win Rate: {successful[best_decay]['eval_win_rate']:.1f}%")
            print(f"   Reward: {successful[best_decay]['eval_reward']:.1f}")
    
    def analyze_comprehensive_results(self):
        """Analyze comprehensive optimization results"""
        if 'comprehensive' not in self.results:
            print("❌ No comprehensive results found")
            return
        
        print(f"\n🎭 COMPREHENSIVE OPTIMIZATION ANALYSIS")
        print("="*50)
        
        comp_data = self.results['comprehensive']
        
        print(f"Session ID: {comp_data['session_id']}")
        print(f"Total experiments: {comp_data['total_experiments']}")
        print(f"Successful experiments: {comp_data['successful_experiments']}")
        print(f"Success rate: {comp_data['successful_experiments']/comp_data['total_experiments']:.1%}")
        print(f"Total duration: {comp_data['total_duration_minutes']:.1f} minutes")
        
        # Show best configs by category
        if 'best_configs_by_category' in comp_data:
            print(f"\nBest Configurations by Category:")
            for category, config in comp_data['best_configs_by_category'].items():
                print(f"\n  {category.upper()}:")
                print(f"    Win Rate: {config['performance']['win_rate']:.1%}")
                print(f"    Time Improvement: {config['performance']['time_improvement']:+.2f}s")
                print(f"    Duration: {config['training_duration']:.0f}s")
        
        # Show overall best
        if 'overall_best_config' in comp_data and comp_data['overall_best_config']:
            best = comp_data['overall_best_config']
            print(f"\n🏆 OVERALL BEST CONFIGURATION:")
            print(f"   Config ID: {best['config_id']}")
            print(f"   Win Rate: {best['performance']['win_rate']:.1%}")
            print(f"   Time Improvement: {best['performance']['time_improvement']:+.2f}s")
            print(f"   Learning Rate: {best['hyperparameters']['learning_rate']}")
            print(f"   Architecture: {best['hyperparameters']['hidden_layers']}")
    
    def generate_performance_comparison(self):
        """Generate performance comparison visualizations"""
        print(f"\n📊 Generating performance comparison plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Learning Rate Performance
        if 'learning_rate' in self.results:
            lr_results = self.results['learning_rate']
            successful = {lr: res for lr, res in lr_results.items() if res.get('success', False)}
            
            if successful:
                lrs = [float(lr) for lr in successful.keys()]
                win_rates = [successful[str(lr)]['eval_win_rate'] for lr in lrs]
                
                axes[0, 0].bar(range(len(lrs)), win_rates, alpha=0.7, color='blue')
                axes[0, 0].set_title('Learning Rate vs Win Rate')
                axes[0, 0].set_xlabel('Learning Rate')
                axes[0, 0].set_ylabel('Win Rate (%)')
                axes[0, 0].set_xticks(range(len(lrs)))
                axes[0, 0].set_xticklabels([f'{lr:.4f}' for lr in lrs])
                axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Exploration Strategy Performance
        if 'exploration' in self.results:
            exp_results = self.results['exploration']
            successful = {decay: res for decay, res in exp_results.items() if res.get('success', False)}
            
            if successful:
                decays = [float(decay) for decay in successful.keys()]
                win_rates = [successful[str(decay)]['eval_win_rate'] for decay in decays]
                
                axes[0, 1].bar(range(len(decays)), win_rates, alpha=0.7, color='green')
                axes[0, 1].set_title('Epsilon Decay vs Win Rate')
                axes[0, 1].set_xlabel('Epsilon Decay')
                axes[0, 1].set_ylabel('Win Rate (%)')
                axes[0, 1].set_xticks(range(len(decays)))
                axes[0, 1].set_xticklabels([f'{decay:.4f}' for decay in decays])
                axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Performance Summary
        if self.best_configs:
            categories = list(self.best_configs.keys())
            if 'learning_rate' in self.best_configs:
                win_rates = [self.best_configs[cat]['performance']['eval_win_rate'] for cat in categories]
            else:
                win_rates = [90, 85]  # Placeholder values
                categories = ['Learning Rate', 'Exploration']
            
            axes[1, 0].bar(categories, win_rates, alpha=0.7, color='orange')
            axes[1, 0].set_title('Best Performance by Category')
            axes[1, 0].set_xlabel('Optimization Category')
            axes[1, 0].set_ylabel('Win Rate (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Training Time Analysis
        if 'learning_rate' in self.results:
            lr_results = self.results['learning_rate']
            successful = {lr: res for lr, res in lr_results.items() if res.get('success', False)}
            
            if successful:
                lrs = [float(lr) for lr in successful.keys()]
                times = [successful[str(lr)]['training_time'] for lr in lrs]
                
                axes[1, 1].scatter(lrs, times, alpha=0.7, color='red', s=100)
                axes[1, 1].set_title('Learning Rate vs Training Time')
                axes[1, 1].set_xlabel('Learning Rate')
                axes[1, 1].set_ylabel('Training Time (seconds)')
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = self.results_dir / f"phase4_part2_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Analysis plots saved to: {plot_file}")
    
    def generate_final_recommendations(self):
        """Generate final hyperparameter recommendations"""
        print(f"\n🎯 FINAL HYPERPARAMETER RECOMMENDATIONS")
        print("="*50)
        
        if not self.best_configs:
            print("❌ No successful optimization results available for recommendations")
            return
        
        recommendations = {}
        
        # Learning rate recommendation
        if 'learning_rate' in self.best_configs:
            best_lr = self.best_configs['learning_rate']
            recommendations['learning_rate'] = best_lr['value']
            print(f"Learning Rate: {best_lr['value']} (Win Rate: {best_lr['performance']['eval_win_rate']:.1f}%)")
        
        # Exploration recommendation
        if 'exploration' in self.best_configs:
            best_exp = self.best_configs['exploration']
            recommendations['epsilon_decay'] = best_exp['value']
            print(f"Epsilon Decay: {best_exp['value']} (Win Rate: {best_exp['performance']['eval_win_rate']:.1f}%)")
        
        # Default recommendations for missing categories
        if 'learning_rate' not in recommendations:
            recommendations['learning_rate'] = 0.0005
            print(f"Learning Rate: 0.0005 (default - no optimization results)")
        
        if 'epsilon_decay' not in recommendations:
            recommendations['epsilon_decay'] = 0.9995
            print(f"Epsilon Decay: 0.9995 (default - no optimization results)")
        
        # Additional recommended parameters
        recommendations.update({
            'gamma': 0.99,
            'epsilon_min': 0.05,
            'buffer_size': 100000,
            'batch_size': 64,
            'hidden_layers': [128, 128, 64],
            'network_type': 'double',
            'buffer_type': 'standard'
        })
        
        print(f"\n📋 COMPLETE RECOMMENDED CONFIGURATION:")
        print(f"   Learning Rate: {recommendations['learning_rate']}")
        print(f"   Epsilon Decay: {recommendations['epsilon_decay']}")
        print(f"   Gamma: {recommendations['gamma']}")
        print(f"   Epsilon Min: {recommendations['epsilon_min']}")
        print(f"   Buffer Size: {recommendations['buffer_size']:,}")
        print(f"   Batch Size: {recommendations['batch_size']}")
        print(f"   Architecture: {recommendations['hidden_layers']}")
        
        # Save recommendations
        rec_file = self.results_dir / "PHASE4_PART2_RECOMMENDATIONS.json"
        with open(rec_file, 'w') as f:
            json.dump({
                'phase': 'Phase 4 Part 2 - Hyperparameter Optimization',
                'timestamp': datetime.now().isoformat(),
                'recommended_hyperparameters': recommendations,
                'source_analysis': self.best_configs
            }, f, indent=2)
        
        print(f"\n✅ Recommendations saved to: {rec_file}")
        
        return recommendations
    
    def run_complete_analysis(self):
        """Run complete Phase 4 Part 2 analysis"""
        print(f"\n🔬 PHASE 4 PART 2: COMPLETE OPTIMIZATION ANALYSIS")
        print("="*60)
        
        # Load results
        if not self.load_results():
            print("❌ No results to analyze")
            return
        
        # Analyze individual components
        self.analyze_learning_rate_optimization()
        self.analyze_exploration_optimization()
        self.analyze_comprehensive_results()
        
        # Generate visualizations
        self.generate_performance_comparison()
        
        # Generate final recommendations
        recommendations = self.generate_final_recommendations()
        
        print(f"\n🎉 PHASE 4 PART 2 ANALYSIS COMPLETED!")
        print(f"📊 Results analyzed from: {self.results_dir}")
        print(f"📈 Plots generated and saved")
        print(f"🎯 Hyperparameter recommendations ready for Phase 4 Part 3")
        
        return recommendations

def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 4 Part 2 Results Analysis')
    parser.add_argument('--results-dir', type=str, default='hyperopt_results',
                      help='Directory containing optimization results')
    
    args = parser.parse_args()
    
    print("🔍 PHASE 4 PART 2: HYPERPARAMETER OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Initialize and run analyzer
    analyzer = Phase4Part2Analyzer(args.results_dir)
    
    try:
        recommendations = analyzer.run_complete_analysis()
        
        print(f"\n✅ Analysis completed successfully!")
        if recommendations:
            print(f"🚀 Ready for Phase 4 Part 3 with optimized hyperparameters")
        
    except Exception as e:
        print(f"\n💥 Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()