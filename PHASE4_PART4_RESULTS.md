# Phase 4 Part 4 – Comprehensive Evaluation (Summary)

- Tournament tool: `tournament.py`
- Default: runs N headless races, saves JSON summary, per‑race stats, and plots.

Quick run example (20 races):
- Win rate: 100.0% (95% CI 83.9–100.0%)
- RL avg time: 11.39s
- Time improvement (vs ~18.83s baseline): mean +7.44s (±1.18s), median +7.33s
- Tile hits (A/D): 294 / 292
- Lane usage (% of steps): lane0 32.0, lane1 39.9, lane2 22.8 (approx)

Files:
- Results JSON: `evaluation_quick/tournament_results_*.json`
- Plots (if available): `evaluation_quick/tournament_plots_*.png`

Run full evaluation:
```bash
python tournament.py --model models/dqn_racing_final.pth --races 100 --out-dir evaluation
```
