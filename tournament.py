#!/usr/bin/env python3
"""
Phase 4 Part 4: Comprehensive Evaluation
- Head-to-head tournament between trained DQN and baseline
- Statistical analysis, strategy snapshots, and simple plots
"""

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.environment.race_environment import RaceEnvironment
from src.agents.dqn_agent import create_racing_dqn_agent


def load_trained_agent(model_path: str):
    config = {
        'state_size': 11,
        'action_size': 5,
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon': 0.0,  # evaluation only
        'epsilon_min': 0.0,
        'epsilon_decay': 1.0,
        'buffer_size': 100000,
        'batch_size': 64,
        'target_update_freq': 1000,
        'network_type': 'double'
    }
    agent = create_racing_dqn_agent(config)
    agent.load_model(model_path, load_optimizer=False)
    return agent


def load_baseline_mean(default: float = 18.83) -> float:
    try:
        files = sorted([f for f in os.listdir('.') if f.startswith('baseline_benchmark_') and f.endswith('.json')])
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


@dataclass
class RaceResult:
    winner: str
    race_time: float
    rl_laps: int
    baseline_laps: int
    rl_accel_hits: int
    rl_decel_hits: int
    rl_lane_usage: Dict[int, int]


def run_tournament(model_path: str, races: int = 100, out_dir: str = 'evaluation', tile_density: float = 0.8,
                   accel_ratio: float = 0.6, angle_bins: int = 36) -> Dict:
    os.makedirs(out_dir, exist_ok=True)
    agent = load_trained_agent(model_path)
    baseline_mean = load_baseline_mean()

    # Aggregates
    results: List[RaceResult] = []
    per_race_records: List[Dict] = []
    rl_times: List[float] = []
    time_improvements: List[float] = []
    wins = {'RL': 0, 'Baseline': 0}
    accel_hits_total = 0
    decel_hits_total = 0
    lane_usage_total = {0: 0, 1: 0, 2: 0}
    # Occupancy heatmap: lanes x angle bins
    occupancy = np.zeros((3, angle_bins), dtype=np.int64)

    for i in range(races):
        env = RaceEnvironment(render_mode=None)
        obs, info = env.reset(seed=5000 + i, options={'tile_density': tile_density, 'accel_ratio': accel_ratio})

        rl_prev_mult = info.get('rl_car_speed_multiplier', 1.0)
        accel_hits = 0
        decel_hits = 0
        lane_usage = {0: 0, 1: 0, 2: 0}
        # Lap timing
        rl_last_laps = int(info.get('rl_car_laps', 0))
        rl_last_lap_time = float(info.get('race_time', 0.0))
        rl_lap_times: List[float] = []

        while True:
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)

            # Lane usage
            lane = int(info.get('rl_car_lane', 0))
            lane_usage[lane] = lane_usage.get(lane, 0) + 1
            lane_usage_total[lane] = lane_usage_total.get(lane, 0) + 1

            # Occupancy by angle bin
            angle = float(info.get('rl_car_angle', 0.0))
            ang_bin = int((angle % (2 * math.pi)) / (2 * math.pi) * angle_bins)
            ang_bin = min(max(0, ang_bin), angle_bins - 1)
            occupancy[lane, ang_bin] += 1

            # Tile hits via speed multiplier transitions
            rl_mult = float(info.get('rl_car_speed_multiplier', 1.0))
            if rl_mult > 1.0 and rl_prev_mult <= 1.0:
                accel_hits += 1
                accel_hits_total += 1
            if rl_mult < 1.0 and rl_prev_mult >= 1.0:
                decel_hits += 1
                decel_hits_total += 1
            rl_prev_mult = rl_mult

            # Lap timing update
            rl_laps_now = int(info.get('rl_car_laps', 0))
            if rl_laps_now > rl_last_laps:
                # Completed a lap
                lap_time = float(info.get('race_time', 0.0)) - rl_last_lap_time
                rl_lap_times.append(lap_time)
                rl_last_lap_time = float(info.get('race_time', 0.0))
                rl_last_laps = rl_laps_now

            if terminated or truncated:
                break

        # Record per-race
        winner = info.get('winner') or 'timeout'
        race_time = float(info.get('race_time', 0.0))
        rl_laps = int(info.get('rl_car_laps', 0))
        base_laps = int(info.get('baseline_car_laps', 0))
        rr = RaceResult(winner, race_time, rl_laps, base_laps, accel_hits, decel_hits, lane_usage)
        results.append(rr)
        rd = asdict(rr)
        rd['rl_lap_times'] = rl_lap_times
        per_race_records.append(rd)
        if winner == 'RL':
            wins['RL'] += 1
        elif winner == 'Baseline':
            wins['Baseline'] += 1

        rl_times.append(race_time)
        time_improvements.append(baseline_mean - race_time)

        env.close()

    # Summary stats
    win_rate = wins['RL'] / races if races > 0 else 0.0
    mean_imp = float(np.mean(time_improvements)) if time_improvements else 0.0
    median_imp = float(np.median(time_improvements)) if time_improvements else 0.0
    std_imp = float(np.std(time_improvements, ddof=1)) if len(time_improvements) > 1 else 0.0

    # Confidence intervals
    # Win rate Wilson interval (95%)
    z = 1.96
    n = races
    if n > 0:
        phat = win_rate
        denom = 1 + z**2 / n
        center = (phat + z**2 / (2*n)) / denom
        radius = z * math.sqrt((phat*(1-phat)/n) + (z**2/(4*n**2))) / denom
        win_rate_ci = (max(0.0, center - radius), min(1.0, center + radius))
    else:
        win_rate_ci = (0.0, 0.0)

    # Mean improvement CI (normal approx)
    if n > 1 and std_imp > 0:
        se = std_imp / math.sqrt(n)
        mean_imp_ci = (mean_imp - z*se, mean_imp + z*se)
    else:
        mean_imp_ci = (mean_imp, mean_imp)

    summary = {
        'model_path': model_path,
        'races': races,
        'tile_density': tile_density,
        'baseline_mean_time': load_baseline_mean(),
        'wins': wins,
        'win_rate': win_rate,
        'win_rate_ci_95': win_rate_ci,
        'rl_avg_time': float(np.mean(rl_times)) if rl_times else None,
        'time_improvement': {
            'mean': mean_imp,
            'median': median_imp,
            'std': std_imp,
            'ci95': mean_imp_ci
        },
        'tile_hits': {
            'acceleration': accel_hits_total,
            'deceleration': decel_hits_total
        },
        'lane_usage_total': lane_usage_total,
        'occupancy_bins': {
            'lanes': 3,
            'angle_bins': angle_bins,
            'matrix': occupancy.tolist()
        },
        'tile_settings': {
            'tile_density': tile_density,
            'accel_ratio': accel_ratio
        }
    }

    # Save per-race and summary
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(out_dir, f'tournament_results_{ts}.json')
    with open(results_path, 'w') as f:
        # Rebuild per_race dicts to include lap times
        json.dump({'summary': summary, 'per_race': per_race_records}, f, indent=2)

    # Plots
    try:
        # 1. Race time distribution
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(rl_times, bins=20, alpha=0.75, color='steelblue')
        plt.axvline(summary['baseline_mean_time'], color='red', linestyle='--', label='Baseline mean')
        plt.title('RL Race Time Distribution')
        plt.xlabel('Seconds')
        plt.ylabel('Count')
        plt.legend()

        # 2. Time improvement distribution
        plt.subplot(2, 2, 2)
        plt.hist(time_improvements, bins=20, alpha=0.75, color='seagreen')
        plt.title('Time Improvement Distribution')
        plt.xlabel('Seconds (baseline - RL)')
        plt.ylabel('Count')

        # 3. Lane usage (overall)
        plt.subplot(2, 2, 3)
        lanes = [0, 1, 2]
        usage = [lane_usage_total.get(l, 0) for l in lanes]
        total_steps = sum(usage) if sum(usage) > 0 else 1
        usage_pct = [u/total_steps*100 for u in usage]
        plt.bar([str(l) for l in lanes], usage_pct, color='orange', alpha=0.8)
        plt.title('Lane Usage (%)')
        plt.xlabel('Lane ID')
        plt.ylabel('Percent of steps')

        # 4. Strategy heatmap (lane x angle bins)
        plt.subplot(2, 2, 4)
        plt.imshow(occupancy, aspect='auto', cmap='viridis')
        plt.title('Strategy Heatmap (Lane x Angle Bins)')
        plt.xlabel('Angle bin (0..2π)')
        plt.ylabel('Lane ID')
        plt.colorbar(label='Step count')

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f'tournament_plots_{ts}.png')
        plt.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close()
        summary['plots'] = {'tournament': plot_path}
    except Exception as e:
        summary['plot_error'] = str(e)

    # Save summary again with plot path
    with open(results_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_race': [asdict(r) for r in results]
        }, f, indent=2)

    # Simple markdown
    md_path = os.path.join(out_dir, f'TOURNAMENT_SUMMARY_{ts}.md')
    with open(md_path, 'w') as f:
        f.write('# Tournament Summary\n\n')
        f.write(f"- Races: {races}\n")
        f.write(f"- Win rate: {win_rate:.1%} (95% CI {win_rate_ci[0]:.1%}–{win_rate_ci[1]:.1%})\n")
        f.write(f"- RL avg time: {summary['rl_avg_time']:.2f}s\n")
        f.write(f"- Time improvement: mean {mean_imp:+.2f}s (±{std_imp:.2f}s), median {median_imp:+.2f}s\n")
        f.write(f"- Tile hits (A/D): {accel_hits_total}/{decel_hits_total}\n")
        f.write(f"- Plots: {summary.get('plots', {}).get('tournament', 'n/a')}\n")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Head-to-Head Tournament Evaluation')
    parser.add_argument('--model', '-m', default='models/dqn_racing_final.pth', help='Path to trained model')
    parser.add_argument('--races', '-r', type=int, default=100, help='Number of races to run')
    parser.add_argument('--out-dir', type=str, default='evaluation', help='Output directory')
    parser.add_argument('--tile-density', type=float, default=0.8, help='Evaluation tile density')
    parser.add_argument('--accel-ratio', type=float, default=0.6, help='Acceleration tile ratio (0..1)')
    parser.add_argument('--quick', action='store_true', help='Quick mode (20 races)')
    args = parser.parse_args()

    if args.quick:
        args.races = min(args.races, 20)

    print('=== Head-to-Head Tournament ===')
    print(f'Model: {args.model}')
    print(f'Races: {args.races}, Tile density: {args.tile_density}')
    print(f'Output: {args.out_dir}')

    summary = run_tournament(args.model, args.races, args.out_dir, args.tile_density, args.accel_ratio)
    print('--- Summary ---')
    print(f"Win rate: {summary['win_rate']:.1%}")
    print(f"Avg RL time: {summary['rl_avg_time']:.2f}s")
    print(f"Time improvement mean: {summary['time_improvement']['mean']:+.2f}s")


if __name__ == '__main__':
    main()
