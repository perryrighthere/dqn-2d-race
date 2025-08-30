#!/usr/bin/env python3
"""
Phase 2: Baseline Agent Benchmarking
Establishes performance benchmarks for the deterministic baseline agent
"""

import sys
import os
import statistics
import json
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__)))

from src.environment.race_environment import RaceEnvironment
import numpy as np

class BaselineBenchmark:
    """Benchmark testing for baseline agent consistency and performance"""
    
    def __init__(self, num_trials: int = 50, max_time_per_race: float = 300.0):
        self.num_trials = num_trials
        self.max_time_per_race = max_time_per_race
        self.results = {
            'completion_times': [],
            'lane_violations': [],
            'lap_times': [],
            'angular_speeds': [],
            'race_outcomes': [],
            'consistency_metrics': {}
        }
        
    def run_single_baseline_race(self, seed: int = None) -> dict:
        """Run a single race with only baseline agent to measure pure performance"""
        env = RaceEnvironment(render_mode=None)
        
        if seed is not None:
            np.random.seed(seed)
            
        _, info = env.reset(seed=seed)
        
        race_data = {
            'completion_time': 0.0,
            'lap_times': [],
            'lane_violations': 0,
            'angular_speeds': [],
            'finished': False,
            'winner': None
        }
        
        last_lap_count = 0
        lap_start_time = 0.0
        step_count = 0
        
        while step_count < self.max_time_per_race * 60:  # 60 FPS
            # Baseline doesn't take actions, but we step with neutral action
            _, _, terminated, _, info = env.step(0)  # Stay action
            
            race_data['completion_time'] = info['race_time']
            race_data['angular_speeds'].append(info['rl_car_angular_speed'])
            
            # Track lap completions
            if info['baseline_car_laps'] > last_lap_count:
                if last_lap_count > 0:  # Not the first lap
                    lap_time = info['race_time'] - lap_start_time
                    race_data['lap_times'].append(lap_time)
                lap_start_time = info['race_time']
                last_lap_count = info['baseline_car_laps']
            
            # Check if baseline car stays in middle lane
            if hasattr(env.baseline_car, 'current_lane'):
                if env.baseline_car.current_lane != env.track.middle_lane_id:
                    race_data['lane_violations'] += 1
            
            if terminated:
                race_data['finished'] = True
                race_data['winner'] = info['winner']
                break
                
            step_count += 1
        
        env.close()
        return race_data
        
    def run_baseline_consistency_test(self, num_tests: int = 10) -> dict:
        """Test baseline agent consistency across multiple runs"""
        print(f"Running baseline consistency test with {num_tests} trials...")
        
        consistency_data = {
            'completion_times': [],
            'total_lane_violations': 0,
            'lap_time_variations': [],
            'speed_consistency': []
        }
        
        for i in range(num_tests):
            print(f"  Trial {i+1}/{num_tests}")
            race_data = self.run_single_baseline_race(seed=42 + i)
            
            if race_data['finished'] and race_data['winner'] == 'Baseline':
                consistency_data['completion_times'].append(race_data['completion_time'])
                consistency_data['total_lane_violations'] += race_data['lane_violations']
                
                if len(race_data['lap_times']) > 1:
                    lap_std = statistics.stdev(race_data['lap_times'])
                    consistency_data['lap_time_variations'].append(lap_std)
                    
                if len(race_data['angular_speeds']) > 0:
                    speed_std = statistics.stdev(race_data['angular_speeds'])
                    consistency_data['speed_consistency'].append(speed_std)
        
        return consistency_data
        
    def run_performance_benchmark(self) -> dict:
        """Run comprehensive baseline performance benchmark"""
        print(f"Running baseline performance benchmark with {self.num_trials} trials...")
        
        benchmark_results = {
            'successful_races': 0,
            'completion_times': [],
            'average_lap_times': [],
            'lane_violations_total': 0,
            'speed_consistency': [],
            'timestamp': datetime.now().isoformat()
        }
        
        for i in range(self.num_trials):
            if i % 10 == 0:
                print(f"  Progress: {i}/{self.num_trials} trials completed")
                
            race_data = self.run_single_baseline_race(seed=100 + i)
            
            if race_data['finished']:
                benchmark_results['successful_races'] += 1
                benchmark_results['completion_times'].append(race_data['completion_time'])
                benchmark_results['lane_violations_total'] += race_data['lane_violations']
                
                if len(race_data['lap_times']) > 0:
                    avg_lap_time = statistics.mean(race_data['lap_times'])
                    benchmark_results['average_lap_times'].append(avg_lap_time)
                    
                if len(race_data['angular_speeds']) > 0:
                    speed_std = statistics.stdev(race_data['angular_speeds'])
                    benchmark_results['speed_consistency'].append(speed_std)
        
        return benchmark_results
        
    def analyze_results(self, results: dict) -> dict:
        """Analyze benchmark results and calculate statistics"""
        analysis = {}
        
        if len(results['completion_times']) > 0:
            analysis['completion_time_stats'] = {
                'mean': statistics.mean(results['completion_times']),
                'median': statistics.median(results['completion_times']),
                'std_dev': statistics.stdev(results['completion_times']) if len(results['completion_times']) > 1 else 0,
                'min': min(results['completion_times']),
                'max': max(results['completion_times']),
                'coefficient_of_variation': statistics.stdev(results['completion_times']) / statistics.mean(results['completion_times']) if len(results['completion_times']) > 1 else 0
            }
            
        if len(results['average_lap_times']) > 0:
            analysis['lap_time_stats'] = {
                'mean': statistics.mean(results['average_lap_times']),
                'median': statistics.median(results['average_lap_times']),
                'std_dev': statistics.stdev(results['average_lap_times']) if len(results['average_lap_times']) > 1 else 0,
                'min': min(results['average_lap_times']),
                'max': max(results['average_lap_times'])
            }
            
        analysis['lane_violation_rate'] = results['lane_violations_total'] / (results['successful_races'] * 180)  # violations per second
        analysis['success_rate'] = results['successful_races'] / self.num_trials
        
        if len(results['speed_consistency']) > 0:
            analysis['speed_consistency'] = {
                'mean_variation': statistics.mean(results['speed_consistency']),
                'max_variation': max(results['speed_consistency'])
            }
            
        return analysis
        
    def generate_report(self, results: dict, analysis: dict) -> str:
        """Generate human-readable benchmark report"""
        report = []
        report.append("=" * 60)
        report.append("BASELINE AGENT BENCHMARK REPORT - PHASE 2")
        report.append("=" * 60)
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Total Trials: {self.num_trials}")
        report.append(f"Successful Races: {results['successful_races']}")
        report.append(f"Success Rate: {analysis['success_rate']:.1%}")
        report.append("")
        
        if 'completion_time_stats' in analysis:
            stats = analysis['completion_time_stats']
            report.append("RACE COMPLETION TIME PERFORMANCE:")
            report.append(f"  Mean: {stats['mean']:.2f} seconds")
            report.append(f"  Median: {stats['median']:.2f} seconds")
            report.append(f"  Std Dev: {stats['std_dev']:.2f} seconds")
            report.append(f"  Range: {stats['min']:.2f} - {stats['max']:.2f} seconds")
            report.append(f"  Coefficient of Variation: {stats['coefficient_of_variation']:.3f}")
            report.append("")
            
        if 'lap_time_stats' in analysis:
            stats = analysis['lap_time_stats']
            report.append("LAP TIME PERFORMANCE:")
            report.append(f"  Mean Lap Time: {stats['mean']:.2f} seconds")
            report.append(f"  Median Lap Time: {stats['median']:.2f} seconds")
            report.append(f"  Lap Time Std Dev: {stats['std_dev']:.2f} seconds")
            report.append(f"  Lap Time Range: {stats['min']:.2f} - {stats['max']:.2f} seconds")
            report.append("")
            
        report.append("CONSISTENCY VALIDATION:")
        report.append(f"  Total Lane Violations: {results['lane_violations_total']}")
        report.append(f"  Lane Violation Rate: {analysis['lane_violation_rate']:.6f} violations/second")
        
        if analysis['lane_violation_rate'] < 0.001:
            report.append("  ✓ PASS: Baseline agent maintains lane discipline")
        else:
            report.append("  ✗ FAIL: Baseline agent shows lane violations")
            
        report.append("")
        
        if 'speed_consistency' in analysis:
            speed_stats = analysis['speed_consistency']
            report.append("SPEED CONSISTENCY:")
            report.append(f"  Mean Speed Variation: {speed_stats['mean_variation']:.6f}")
            report.append(f"  Max Speed Variation: {speed_stats['max_variation']:.6f}")
            
            if speed_stats['mean_variation'] < 0.01:
                report.append("  ✓ PASS: Baseline agent maintains consistent speed")
            else:
                report.append("  ✗ FAIL: Baseline agent shows speed inconsistency")
                
        report.append("")
        report.append("BENCHMARK SUMMARY:")
        if analysis.get('success_rate', 0) > 0.95 and analysis.get('lane_violation_rate', 1) < 0.001:
            report.append("  ✓ PHASE 2 COMPLETE: Baseline agent meets performance requirements")
            report.append("  ✓ Ready for Phase 3: DQN Agent Development")
        else:
            report.append("  ✗ PHASE 2 INCOMPLETE: Baseline agent needs improvements")
            
        report.append("=" * 60)
        
        return "\n".join(report)
        
    def save_results(self, results: dict, analysis: dict, filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"baseline_benchmark_{timestamp}.json"
            
        output_data = {
            'benchmark_results': results,
            'statistical_analysis': analysis,
            'metadata': {
                'num_trials': self.num_trials,
                'max_time_per_race': self.max_time_per_race,
                'generated_at': datetime.now().isoformat()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
            
        return filename

def main():
    print("Phase 2: Baseline Agent Benchmarking System")
    print("==========================================")
    
    # Create benchmark instance
    benchmark = BaselineBenchmark(num_trials=30)  # Reduced for faster testing
    
    try:
        # Run consistency test first
        consistency_results = benchmark.run_baseline_consistency_test(num_tests=10)
        
        print("\nConsistency Test Results:")
        if len(consistency_results['completion_times']) > 0:
            mean_time = statistics.mean(consistency_results['completion_times'])
            std_time = statistics.stdev(consistency_results['completion_times']) if len(consistency_results['completion_times']) > 1 else 0
            print(f"  Mean completion time: {mean_time:.2f}s (±{std_time:.2f}s)")
        
        print(f"  Total lane violations: {consistency_results['total_lane_violations']}")
        
        # Run full performance benchmark
        print("\nRunning full performance benchmark...")
        results = benchmark.run_performance_benchmark()
        
        # Analyze results
        analysis = benchmark.analyze_results(results)
        
        # Generate and display report
        report = benchmark.generate_report(results, analysis)
        print("\n" + report)
        
        # Save results
        filename = benchmark.save_results(results, analysis)
        print(f"\nDetailed results saved to: {filename}")
        
        # Create summary for phase completion documentation
        summary_filename = "PHASE2_BASELINE_BENCHMARKS.md"
        with open(summary_filename, 'w') as f:
            f.write("# Phase 2 Completion: Baseline Agent Benchmarks\n\n")
            f.write("## Performance Summary\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            if 'completion_time_stats' in analysis:
                stats = analysis['completion_time_stats']
                f.write("### Race Completion Benchmarks\n")
                f.write(f"- **Mean Race Time**: {stats['mean']:.2f} seconds\n")
                f.write(f"- **Standard Deviation**: {stats['std_dev']:.2f} seconds\n")
                f.write(f"- **Consistency (CV)**: {stats['coefficient_of_variation']:.3f}\n\n")
                
            f.write("### Baseline Agent Validation\n")
            f.write(f"- **Success Rate**: {analysis['success_rate']:.1%}\n")
            f.write(f"- **Lane Violations**: {results['lane_violations_total']} total\n")
            f.write(f"- **Lane Discipline**: {'✓ PASS' if analysis['lane_violation_rate'] < 0.001 else '✗ FAIL'}\n\n")
            
            f.write("### Phase 2 Status\n")
            if analysis.get('success_rate', 0) > 0.95 and analysis.get('lane_violation_rate', 1) < 0.001:
                f.write("✅ **PHASE 2 COMPLETE** - Baseline agent benchmarks established\n\n")
                f.write("**Ready for Phase 3**: DQN Agent Development\n")
            else:
                f.write("❌ **PHASE 2 INCOMPLETE** - Baseline agent requires improvements\n")
            
        print(f"Phase 2 summary saved to: {summary_filename}")
        
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()