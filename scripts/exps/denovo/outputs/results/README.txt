De novo sampling experiment results — 2026-03-05
======================================================

Total: 219 runs, 8153 molecules (SMILES)

1. hp_sweep/  (194 configs)
   Full hyperparameter sweep at matched rollout budgets.
   Source scripts: sweep_full_hp.sh, sweep_noelite_divpenalty.sh

   Naming convention:
     beam_B{budget}_N{beam_width}_L{branching}_t{temp}
     beam_noelite_B{budget}_N{width}_L{branching}_t{temp}
     beam_div{lambda}_B{budget}_N{width}_L{branching}_t{temp}
     mcts_B{budget}_L{branching}_c{c_uct}
     mcts_ref_B{budget}_L{branching}_c{c_uct}
     uncond_baseline

2. budget_curves/  (25 configs)
   Quality vs compute budget at fixed L=4, N=50.
   Source script: sweep_budget.sh

   Naming convention:
     budget_K{steps}_L{branching}_{variant}
     variant ∈ {default (temp=0.8), t05 (temp=0.5), div (diversity), rand (randomness)}
     budget_standard = unconditional baseline

3. boltz_eval/
   Boltz structure prediction on selected molecules.
   Source script: run_boltz_eval.py
