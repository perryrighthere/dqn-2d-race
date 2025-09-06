#!/usr/bin/env python3
"""
Create comprehensive analysis of Phase 4 Part 2 optimization results
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

def load_optimization_results(results_dir: str = "hyperopt_results") -> Dict[str, List[Dict]]:
    """Load all optimization results by category"""
    results_by_category = {
        'learning_rate': [],
        'architecture': [],
        'exploration': [],
        'replay': [],
        'gamma': []
    }
    
    logs_dir = Path(results_dir) / "logs"
    
    # Map file patterns to categories
    patterns = {
        'lr_*.json': 'learning_rate',
        'arch_*.json': 'architecture', 
        'explore_*.json': 'exploration',
        'replay_*.json': 'replay',
        'gamma_*.json': 'gamma'
    }
    
    for pattern, category in patterns.items():
        files = glob.glob(str(logs_dir / pattern))
        for file_path in files:
            if 'training_history' not in file_path:
                try:
                    with open(file_path, 'r') as f:
                        result = json.load(f)
                        if result.get('success'):
                            results_by_category[category].append(result)
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return results_by_category

def analyze_category(category_name: str, results: List[Dict]) -> Dict:
    """Analyze results for a specific category"""
    if not results:
        return {'category': category_name, 'experiments': 0, 'best_result': None}
    
    print(f"\n📊 {category_name.upper()} OPTIMIZATION RESULTS")
    print("="*50)
    
    best_result = None
    best_score = -1
    
    for result in results:
        final_results = result['final_results']
        win_rate = final_results['win_rate']
        time_improvement = final_results['time_improvement']
        
        # Score function: win rate most important, time improvement secondary
        score = win_rate * 1000 + max(0, time_improvement) * 10
        
        # Extract key parameter
        hyperparams = result['hyperparameters']
        if category_name == 'learning_rate':
            key_param = f"LR {hyperparams['learning_rate']}"
        elif category_name == 'architecture':
            key_param = f"Arch {hyperparams['hidden_layers']}"
        elif category_name == 'exploration':
            key_param = f"Decay {hyperparams['epsilon_decay']}"
        elif category_name == 'replay':
            key_param = f"Buffer {hyperparams['buffer_size']//1000}k/Batch {hyperparams['batch_size']}"
        elif category_name == 'gamma':
            key_param = f"Gamma {hyperparams['gamma']}"
        else:
            key_param = "Unknown"
        
        print(f"  {key_param:<25}: Win {win_rate:.1%}, Time {time_improvement:+6.1f}s, Score {score:.0f}")
        
        if score > best_score:
            best_score = score
            best_result = result
    
    if best_result:
        print(f"\n🏆 BEST {category_name.upper()}:")
        final_results = best_result['final_results']
        hyperparams = best_result['hyperparameters']
        print(f"   Win Rate: {final_results['win_rate']:.1%}")
        print(f"   Race Time: {final_results['avg_race_time']:.2f}s")
        print(f"   Time Improvement: {final_results['time_improvement']:+.2f}s")
        print(f"   Training Duration: {best_result['training_duration']:.0f}s")
        
        # Print key hyperparameter
        if category_name == 'learning_rate':
            print(f"   Learning Rate: {hyperparams['learning_rate']}")
        elif category_name == 'architecture':
            print(f"   Architecture: {hyperparams['hidden_layers']}")
        elif category_name == 'exploration':
            print(f"   Epsilon Decay: {hyperparams['epsilon_decay']}")
            print(f"   Epsilon Min: {hyperparams['epsilon_min']}")
        elif category_name == 'replay':
            print(f"   Buffer Size: {hyperparams['buffer_size']:,}")
            print(f"   Batch Size: {hyperparams['batch_size']}")
        elif category_name == 'gamma':
            print(f"   Gamma: {hyperparams['gamma']}")
    
    return {
        'category': category_name,
        'experiments': len(results),
        'best_result': best_result,
        'all_results': results
    }

def create_final_recommendations(analysis_results: Dict) -> Dict:
    """Create final hyperparameter recommendations"""
    recommendations = {
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.9995,
        'buffer_size': 100000,
        'batch_size': 64,
        'hidden_layers': [128, 128, 64],
        'network_type': 'double',
        'buffer_type': 'standard'
    }
    
    print(f"\n🎯 FINAL HYPERPARAMETER RECOMMENDATIONS")
    print("="*50)
    
    # Update with best results from each category
    for category, analysis in analysis_results.items():
        if analysis['best_result']:
            best_hyperparams = analysis['best_result']['hyperparameters']
            
            if category == 'learning_rate':
                recommendations['learning_rate'] = best_hyperparams['learning_rate']
                print(f"Learning Rate: {recommendations['learning_rate']} (from {category} optimization)")
                
            elif category == 'architecture':
                recommendations['hidden_layers'] = best_hyperparams['hidden_layers']
                print(f"Architecture: {recommendations['hidden_layers']} (from {category} optimization)")
                
            elif category == 'exploration':
                recommendations['epsilon_decay'] = best_hyperparams['epsilon_decay']
                recommendations['epsilon_min'] = best_hyperparams['epsilon_min']
                print(f"Epsilon Decay: {recommendations['epsilon_decay']} (from {category} optimization)")
                print(f"Epsilon Min: {recommendations['epsilon_min']} (from {category} optimization)")
                
            elif category == 'replay':
                recommendations['buffer_size'] = best_hyperparams['buffer_size']
                recommendations['batch_size'] = best_hyperparams['batch_size']
                print(f"Buffer Size: {recommendations['buffer_size']:,} (from {category} optimization)")
                print(f"Batch Size: {recommendations['batch_size']} (from {category} optimization)")
                
            elif category == 'gamma':
                recommendations['gamma'] = best_hyperparams['gamma']
                print(f"Gamma: {recommendations['gamma']} (from {category} optimization)")
    
    return recommendations

def main():
    """Main analysis function"""
    print("🔬 PHASE 4 PART 2: COMPREHENSIVE OPTIMIZATION ANALYSIS")
    print("="*60)
    
    # Load all results
    results_by_category = load_optimization_results()
    
    total_experiments = sum(len(results) for results in results_by_category.values())
    successful_experiments = sum(
        len([r for r in results if r.get('success', False)]) 
        for results in results_by_category.values()
    )
    
    print(f"\n📋 OPTIMIZATION SUMMARY")
    print(f"Total experiments completed: {total_experiments}")
    print(f"Successful experiments: {successful_experiments}")
    print(f"Success rate: {successful_experiments/max(1, total_experiments):.1%}")
    
    # Analyze each category
    analysis_results = {}
    for category_name, results in results_by_category.items():
        if results:  # Only analyze categories with results
            analysis_results[category_name] = analyze_category(category_name, results)
    
    # Create final recommendations
    recommendations = create_final_recommendations(analysis_results)
    
    # Find overall best configuration
    all_results = []
    for results in results_by_category.values():
        all_results.extend(results)
    
    if all_results:
        best_overall = max(all_results, key=lambda x: x['final_results']['win_rate'] * 1000 + 
                                                     max(0, x['final_results']['time_improvement']) * 10)
        
        print(f"\n🏆 OVERALL BEST CONFIGURATION:")
        print(f"   Config ID: {best_overall['config_id']}")
        print(f"   Category: {best_overall['category']}")
        print(f"   Win Rate: {best_overall['final_results']['win_rate']:.1%}")
        print(f"   Race Time: {best_overall['final_results']['avg_race_time']:.2f}s")
        print(f"   Time Improvement: {best_overall['final_results']['time_improvement']:+.2f}s")
        print(f"   Training Duration: {best_overall['training_duration']:.0f}s")
    
    # Save comprehensive report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_data = {
        'phase': 'Phase 4 Part 2 - Comprehensive Hyperparameter Optimization',
        'timestamp': datetime.now().isoformat(),
        'session_analysis': f'manual_analysis_{timestamp}',
        'total_experiments': total_experiments,
        'successful_experiments': successful_experiments,
        'success_rate': successful_experiments/max(1, total_experiments),
        'analysis_by_category': analysis_results,
        'final_recommendations': recommendations,
        'overall_best_configuration': best_overall if all_results else None
    }
    
    report_file = f"hyperopt_results/COMPREHENSIVE_HYPEROPT_REPORT_{timestamp}.json"
    with open(report_file, 'w') as f:
        json.dump(report_data, f, indent=2)
    
    print(f"\n✅ ANALYSIS COMPLETED!")
    print(f"📊 Comprehensive report saved to: {report_file}")
    print(f"🚀 Ready for Phase 4 Part 3 with optimized hyperparameters!")
    
    return report_data

if __name__ == "__main__":
    main()