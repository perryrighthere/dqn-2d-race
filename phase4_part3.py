#!/usr/bin/env python3
"""
Phase 4 Part 3: Advanced Training Pipeline
- Extended deep training with curriculum
- Multi-seed statistical training
- Simple stability analysis and summary report
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict

from train_dqn import TrainingManager, create_training_config


def build_part3_config(episodes: int, output_dir: str, seed: int, use_curriculum: bool = True) -> Dict:
    config = create_training_config("deep")
    config['num_episodes'] = episodes
    config['save_freq'] = 250
    config['eval_freq'] = 50
    config['seed'] = seed
    config['plot_training'] = True
    config['render_training'] = False

    # Ensure optimized hyperparameters are present
    config['agent_config'].update({
        'learning_rate': 0.0005,
        'gamma': 0.99,
        'epsilon_min': 0.05,
        'epsilon_decay': 0.9995,
        'buffer_size': 100000,
        'batch_size': 64,
        'network_type': 'double',
        'buffer_type': 'standard'
    })

    # Curriculum schedule
    config['curriculum']['enabled'] = bool(use_curriculum)
    if use_curriculum and not config['curriculum'].get('stages'):
        config['curriculum']['stages'] = [
            {'start_episode': 1, 'tile_density': 0.5},
            {'start_episode': 801, 'tile_density': 0.8},
            {'start_episode': 1601, 'tile_density': 1.1}
        ]

    # Outputs
    seed_dir = os.path.join(output_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    config['save_dir'] = seed_dir
    config['log_dir'] = os.path.join(output_dir, 'logs')
    os.makedirs(config['log_dir'], exist_ok=True)

    return config


def run_single_training(config: Dict) -> Dict:
    trainer = TrainingManager(config)
    trainer.setup()
    final_results = trainer.train()
    trainer.cleanup()
    return {
        'final': final_results,
        'evaluation_history': trainer.training_history.get('evaluation_results', [])
    }


def summarize_results(results_by_seed: Dict[int, Dict]) -> Dict:
    # Aggregate final results
    win_rates = []
    improvements = []
    avg_times = []
    for seed, data in results_by_seed.items():
        final = data['final']
        win_rates.append(final['win_rate'])
        improvements.append(final['time_improvement'])
        avg_times.append(final['avg_race_time'])

    summary = {
        'seeds': list(results_by_seed.keys()),
        'final_win_rate_mean': float(sum(win_rates) / len(win_rates)) if win_rates else 0.0,
        'final_time_improvement_mean': float(sum(improvements) / len(improvements)) if improvements else 0.0,
        'final_avg_time_mean': float(sum(avg_times) / len(avg_times)) if avg_times else 0.0,
    }

    # Simple stability: std of last 3 eval win_rates per seed
    stability = {}
    for seed, data in results_by_seed.items():
        hist = data.get('evaluation_history', [])
        last3 = hist[-3:]
        if len(last3) >= 2:
            wrs = [e['win_rate'] for e in last3]
            mean_wr = sum(wrs) / len(wrs)
            var = sum((w - mean_wr) ** 2 for w in wrs) / len(wrs)
            stability[seed] = {'mean_win_rate_last3': mean_wr, 'var_win_rate_last3': var}
        else:
            stability[seed] = {'mean_win_rate_last3': None, 'var_win_rate_last3': None}
    summary['stability'] = stability

    return summary


def write_simple_report(output_dir: str, episodes: int, summary: Dict, best_seed: int):
    md_path = os.path.join(output_dir, 'PHASE4_PART3_RESULTS.md')
    lines = []
    lines.append('# Phase 4 Part 3 Results')
    lines.append('')
    lines.append(f'- Episodes per run: {episodes}')
    lines.append(f'- Seeds: {", ".join(map(str, summary["seeds"]))}')
    lines.append(f'- Final win rate (mean): {summary["final_win_rate_mean"]:.1%}')
    lines.append(f'- Time improvement (mean): {summary["final_time_improvement_mean"]:+.2f}s')
    lines.append(f'- Avg DQN race time (mean): {summary["final_avg_time_mean"]:.2f}s')
    lines.append(f'- Best model (by win rate/time): seed {best_seed} -> {os.path.join(output_dir, f"seed_{best_seed}", "dqn_racing_final.pth")}')
    lines.append('')
    lines.append('Simple stability (last 3 evals per seed):')
    for seed in summary['seeds']:
        stab = summary['stability'].get(seed, {})
        lines.append(f'- Seed {seed}: mean wr={stab.get("mean_win_rate_last3")}, var={stab.get("var_win_rate_last3")}')
    with open(md_path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Phase 4 Part 3: Advanced Training Pipeline')
    parser.add_argument('--episodes', type=int, default=2500, help='Episodes per training run')
    parser.add_argument('--seeds', type=str, default='101,202,303,404,505', help='Comma-separated seeds')
    parser.add_argument('--out-dir', type=str, default='phase4_part3', help='Output directory')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    parser.add_argument('--quick', action='store_true', help='Quick run (reduced episodes)')
    args = parser.parse_args()

    episodes = args.episodes
    if args.quick:
        episodes = min(episodes, 300)

    seeds: List[int] = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    print('=== Phase 4 Part 3: Advanced Training ===')
    print(f'Episodes per seed: {episodes} (quick={args.quick})')
    print(f'Seeds: {seeds}')
    print(f'Output: {out_dir}')

    results_by_seed: Dict[int, Dict] = {}
    start = time.time()
    for seed in seeds:
        print(f"\n--- Training seed {seed} ---")
        cfg = build_part3_config(episodes=episodes, output_dir=out_dir, seed=seed, use_curriculum=(not args.no_curriculum))
        res = run_single_training(cfg)
        results_by_seed[seed] = res

    # Aggregate and choose best by win rate then time improvement
    best_seed = max(results_by_seed.keys(), key=lambda s: (results_by_seed[s]['final']['win_rate'], results_by_seed[s]['final']['time_improvement']))
    summary = summarize_results(results_by_seed)
    summary['best_seed'] = best_seed
    summary['completed_at'] = datetime.now().isoformat()

    # Save JSON summary
    json_path = os.path.join(out_dir, f'phase4_part3_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {json_path}")

    # Simple markdown report
    write_simple_report(out_dir, episodes, summary, best_seed)
    print(f"Simple report written to {os.path.join(out_dir, 'PHASE4_PART3_RESULTS.md')}")
    print(f"Total duration: {time.time() - start:.0f}s")


if __name__ == '__main__':
    main()

