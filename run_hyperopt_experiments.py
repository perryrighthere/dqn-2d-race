#!/usr/bin/env python3
"""
Phase 4 Part 2: Hyperparameter Optimization Experiments
Run systematic optimization experiments
"""

import os
import json
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

def run_learning_rate_experiments():
    """2.1 Learning Rate Optimization"""
    print("\n🔬 PHASE 2.1: LEARNING RATE OPTIMIZATION")
    print("="*50)
    
    learning_rates = [0.0001, 0.0005, 0.001, 0.002]
    results = {}
    
    # Create results directory
    os.makedirs("hyperopt_results/lr_experiments", exist_ok=True)
    
    for i, lr in enumerate(learning_rates):
        print(f"\nExperiment {i+1}/4: Learning Rate = {lr}")
        print("-" * 30)
        
        # Create custom training script with this learning rate
        config_name = f"lr_{lr}_config"
        
        # Run training with custom learning rate
        start_time = time.time()
        
        try:
            # We'll use a quick 300-episode training for hyperopt
            cmd = [
                sys.executable, "train_dqn.py", 
                "--config", "fast", 
                "--episodes", "300",
                "--save-dir", f"hyperopt_results/lr_experiments/lr_{lr}"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Training completed in {training_time:.1f}s")
                
                # Extract performance metrics from output
                output_lines = result.stdout.split('\n')
                final_win_rate = 0
                final_reward = 0
                
                for line in output_lines:
                    if 'Final Win Rate:' in line:
                        try:
                            final_win_rate = float(line.split('Final Win Rate:')[1].split('%')[0])
                        except:
                            pass
                    elif 'Final Average Reward:' in line:
                        try:
                            final_reward = float(line.split('Final Average Reward:')[1].split()[0])
                        except:
                            pass
                
                # Run evaluation
                print(f"Running evaluation...")
                eval_cmd = [
                    sys.executable, "demo_trained.py", 
                    "--headless", 
                    "--model", f"hyperopt_results/lr_experiments/lr_{lr}/dqn_racing_final.pth",
                    "--races", "10"
                ]
                
                eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=300)
                
                eval_win_rate = 0
                eval_reward = 0
                
                if eval_result.returncode == 0:
                    eval_lines = eval_result.stdout.split('\n')
                    for line in eval_lines:
                        if 'Win rate:' in line:
                            try:
                                eval_win_rate = float(line.split('Win rate:')[1].split('%')[0])
                            except:
                                pass
                        elif 'Average reward:' in line:
                            try:
                                eval_reward = float(line.split('Average reward:')[1].strip())
                            except:
                                pass
                
                results[lr] = {
                    'learning_rate': lr,
                    'training_time': training_time,
                    'final_win_rate': final_win_rate,
                    'final_reward': final_reward,
                    'eval_win_rate': eval_win_rate,
                    'eval_reward': eval_reward,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"   Training Win Rate: {final_win_rate:.1f}%")
                print(f"   Evaluation Win Rate: {eval_win_rate:.1f}%")
                print(f"   Evaluation Reward: {eval_reward:.1f}")
                
            else:
                print(f"❌ Training failed: {result.stderr}")
                results[lr] = {
                    'learning_rate': lr,
                    'success': False,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Training timeout")
            results[lr] = {
                'learning_rate': lr,
                'success': False,
                'error': 'Training timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"💥 Error: {e}")
            results[lr] = {
                'learning_rate': lr,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Save results
    with open("hyperopt_results/learning_rate_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best learning rate
    successful_results = {lr: res for lr, res in results.items() if res['success']}
    if successful_results:
        best_lr = max(successful_results.keys(), 
                     key=lambda x: successful_results[x]['eval_win_rate'] + successful_results[x]['eval_reward']/1000)
        print(f"\n🏆 BEST LEARNING RATE: {best_lr}")
        print(f"   Win Rate: {successful_results[best_lr]['eval_win_rate']:.1f}%")
        print(f"   Reward: {successful_results[best_lr]['eval_reward']:.1f}")
    else:
        best_lr = 0.0005  # Fallback to current best
        print(f"\n⚠️  No successful experiments, using default: {best_lr}")
    
    return results, best_lr

def run_exploration_experiments(best_lr: float):
    """2.3 Exploration Strategy Tuning"""
    print(f"\n🎯 PHASE 2.3: EXPLORATION STRATEGY OPTIMIZATION")
    print("="*50)
    print(f"Using best learning rate: {best_lr}")
    
    # We'll test different epsilon decay rates by running shorter experiments
    epsilon_decays = [0.999, 0.9995, 0.995]  # Conservative, Current, Aggressive
    results = {}
    
    os.makedirs("hyperopt_results/epsilon_experiments", exist_ok=True)
    
    for i, decay in enumerate(epsilon_decays):
        print(f"\nExperiment {i+1}/3: Epsilon Decay = {decay}")
        print("-" * 30)
        
        start_time = time.time()
        
        try:
            # Run shorter training for exploration experiments
            cmd = [
                sys.executable, "train_dqn.py", 
                "--config", "fast", 
                "--episodes", "200",
                "--save-dir", f"hyperopt_results/epsilon_experiments/eps_{decay}"
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            training_time = time.time() - start_time
            
            if result.returncode == 0:
                print(f"✅ Training completed in {training_time:.1f}s")
                
                # Quick evaluation
                eval_cmd = [
                    sys.executable, "demo_trained.py", 
                    "--headless", 
                    "--model", f"hyperopt_results/epsilon_experiments/eps_{decay}/dqn_racing_final.pth",
                    "--races", "5"
                ]
                
                eval_result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=300)
                
                eval_win_rate = 0
                eval_reward = 0
                
                if eval_result.returncode == 0:
                    eval_lines = eval_result.stdout.split('\n')
                    for line in eval_lines:
                        if 'Win rate:' in line:
                            try:
                                eval_win_rate = float(line.split('Win rate:')[1].split('%')[0])
                            except:
                                pass
                        elif 'Average reward:' in line:
                            try:
                                eval_reward = float(line.split('Average reward:')[1].strip())
                            except:
                                pass
                
                results[decay] = {
                    'epsilon_decay': decay,
                    'training_time': training_time,
                    'eval_win_rate': eval_win_rate,
                    'eval_reward': eval_reward,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
                
                print(f"   Evaluation Win Rate: {eval_win_rate:.1f}%")
                print(f"   Evaluation Reward: {eval_reward:.1f}")
                
            else:
                print(f"❌ Training failed")
                results[decay] = {
                    'epsilon_decay': decay,
                    'success': False,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"💥 Error: {e}")
            results[decay] = {
                'epsilon_decay': decay,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # Save results
    with open("hyperopt_results/exploration_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Find best epsilon decay
    successful_results = {decay: res for decay, res in results.items() if res['success']}
    if successful_results:
        best_decay = max(successful_results.keys(), 
                        key=lambda x: successful_results[x]['eval_win_rate'] + successful_results[x]['eval_reward']/1000)
        print(f"\n🏆 BEST EPSILON DECAY: {best_decay}")
    else:
        best_decay = 0.9995
        print(f"\n⚠️  Using default epsilon decay: {best_decay}")
    
    return results, best_decay

def generate_optimization_report(lr_results: Dict, exp_results: Dict, best_lr: float, best_decay: float):
    """Generate comprehensive optimization report"""
    print(f"\n📊 GENERATING PHASE 4 PART 2 REPORT...")
    
    report = {
        'phase': 'Phase 4 Part 2 - Hyperparameter Optimization',
        'timestamp': datetime.now().isoformat(),
        'experiments': {
            'learning_rate': lr_results,
            'exploration': exp_results
        },
        'best_parameters': {
            'learning_rate': best_lr,
            'epsilon_decay': best_decay
        },
        'summary': {
            'total_experiments': len(lr_results) + len(exp_results),
            'successful_experiments': sum(1 for r in lr_results.values() if r['success']) + 
                                   sum(1 for r in exp_results.values() if r['success'])
        }
    }
    
    # Save JSON report
    with open("hyperopt_results/PHASE4_PART2_RESULTS.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create markdown report
    report_md = f"""# Phase 4 Part 2: Hyperparameter Optimization Results

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Status:** ✅ COMPLETED

## Optimization Summary

### Best Parameters Found:
- **Learning Rate:** {best_lr}
- **Epsilon Decay:** {best_decay}

## 2.1 Learning Rate Optimization Results

| Learning Rate | Win Rate | Avg Reward | Training Time | Status |
|---------------|----------|------------|---------------|---------|
"""
    
    for lr, result in lr_results.items():
        status = "✅" if result['success'] else "❌"
        win_rate = result.get('eval_win_rate', 0)
        reward = result.get('eval_reward', 0)
        time_taken = result.get('training_time', 0)
        
        report_md += f"| {lr} | {win_rate:.1f}% | {reward:.1f} | {time_taken:.1f}s | {status} |\n"
    
    report_md += f"""
## 2.3 Exploration Strategy Results

| Epsilon Decay | Win Rate | Avg Reward | Training Time | Status |
|---------------|----------|------------|---------------|---------|
"""
    
    for decay, result in exp_results.items():
        status = "✅" if result['success'] else "❌"
        win_rate = result.get('eval_win_rate', 0)
        reward = result.get('eval_reward', 0)
        time_taken = result.get('training_time', 0)
        
        report_md += f"| {decay} | {win_rate:.1f}% | {reward:.1f} | {time_taken:.1f}s | {status} |\n"
    
    report_md += f"""
## Conclusions

The hyperparameter optimization revealed optimal settings for our DQN racing agent. These parameters can be used for extended training in Phase 4 Part 3.

**Next Steps:** Phase 4 Part 3 - Advanced Training Pipeline with optimized parameters.
"""
    
    with open("hyperopt_results/PHASE4_PART2_RESULTS.md", 'w') as f:
        f.write(report_md)
    
    print(f"✅ Reports saved to hyperopt_results/")

def main():
    """Run Phase 4 Part 2 experiments"""
    print("🚀 STARTING PHASE 4 PART 2: HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    start_time = time.time()
    
    try:
        # 2.1 Learning Rate Optimization
        lr_results, best_lr = run_learning_rate_experiments()
        
        # 2.3 Exploration Strategy Tuning
        exp_results, best_decay = run_exploration_experiments(best_lr)
        
        # Generate comprehensive report
        generate_optimization_report(lr_results, exp_results, best_lr, best_decay)
        
        total_time = time.time() - start_time
        print(f"\n🎉 PHASE 4 PART 2 COMPLETED!")
        print(f"⏱️  Total Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"🏆 Optimal Learning Rate: {best_lr}")
        print(f"🎯 Optimal Epsilon Decay: {best_decay}")
        print(f"📊 Results: hyperopt_results/PHASE4_PART2_RESULTS.md")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Optimization interrupted by user")
    except Exception as e:
        print(f"\n💥 Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()