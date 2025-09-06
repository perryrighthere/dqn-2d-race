# Phase 4 Part 3 – Progress

- Extended training supported: `deep` preset now 2500 episodes, saving every 250, eval every 50.
- Curriculum enabled: tile density stages 0.5 → 0.8 → 1.1 (adaptive advance if recent win rate ≥ 0.9).
- Multi‑seed pipeline added: `phase4_part3.py` runs N seeds and writes a simple summary.

Quick verification run (seed 101, 50–100 episodes):
- Final win rate: 100%
- Avg DQN race time: ~12.2s
- Time improvement vs baseline (~18.83s): ~+6.6s
- Artifacts: models under `phase4_part3/seed_101/`, logs and plots under `phase4_part3/logs/`.

Next steps:
- Full run: `python phase4_part3.py --episodes 2500 --seeds 101,202,303,404,505`
- Summary: see `phase4_part3/PHASE4_PART3_RESULTS.md` after completion.
