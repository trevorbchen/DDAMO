"""
VIDD fine-tuning runner.

VIDD (Wang et al., arXiv 2507.00445) is a simulation-based, non-differentiable
reward finetuning method for discrete diffusion. This script runs VIDD on
GenMol with FA or Boltz oracles, producing directly comparable results to
DDPP-LB under a fixed oracle budget.

Usage:
    python scripts/run_vidd.py \
        --model_path model_v2.ckpt \
        --oracle fa \
        --loss_func kl \
        --max_oracle_calls 800 \
        --gpu 0 --seed 0 \
        --output_dir outputs/vidd_fa_kl_s0
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "FlashAffinity", "src"))


def build_oracle(args):
    """Build the non-differentiable oracle reward function.

    Returns a callable: list[str] → torch.Tensor[N] (or list[float]).
    """
    import torch

    if args.oracle == "boltz":
        from genmol.rewards.boltz import BoltzAffinityReward
        oracle_model = BoltzAffinityReward(
            cache_dir=args.boltz_cache,
            gpu_id=args.gpu,
            diffusion_samples=args.diffusion_samples,
        )

        def oracle_fn(smiles_list):
            scores = oracle_model(smiles_list)
            if isinstance(scores, torch.Tensor):
                return scores
            return torch.tensor(list(scores), dtype=torch.float32)

    elif args.oracle == "fa":
        from genmol.rewards.flash_affinity import FlashAffinityForwardOp
        fa_model = FlashAffinityForwardOp(
            protein_pdb=args.protein_pdb,
            protein_repr_path=args.protein_repr,
            protein_id=args.protein_id,
            checkpoint_paths=[args.fa_checkpoint],
            task="value",
        )

        def oracle_fn(smiles_list):
            t = fa_model(smiles_list)
            if not isinstance(t, torch.Tensor):
                t = torch.tensor(list(t), dtype=torch.float32)
            # Replace 0.0 (failure sentinel) with -inf so VIDD filters them
            t = torch.where(t == 0.0, torch.tensor(float("-inf")), t)
            return t

    elif args.oracle == "qed":
        from genmol.rewards import get_reward
        fn = get_reward("qed")

        def oracle_fn(smiles_list):
            r = fn(smiles_list)
            return r if isinstance(r, torch.Tensor) else torch.tensor(r, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown oracle: {args.oracle}")

    return oracle_fn


def main():
    parser = argparse.ArgumentParser()
    # === Model / oracle ===
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--oracle", choices=["fa", "boltz", "qed"], default="fa")
    parser.add_argument("--oracle_module", default=None,
                        help="Python import path of module that registers custom oracles")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)

    # === Budget ===
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--wall_budget_sec", type=float, default=None)
    parser.add_argument("--max_oracle_calls", type=int, default=None)

    # === Shared with DDPP ===
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--replay_buffer_size", type=int, default=10_000)
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--min_add_len", type=int, default=60)
    parser.add_argument("--softmax_temp", type=float, default=0.8)
    parser.add_argument("--randomness", type=float, default=0.5)

    # === VIDD-specific ===
    # Defaults: rw_mle + gkd_lmbda=0 → matches DDPP-LB's oracle-call pattern
    # (no per-step oracle calls; all calls come from initial buffer fill).
    # Use --loss_func kl --gkd_lmbda 0.5 for the expensive simulation-based variant.
    parser.add_argument("--loss_func", choices=["rw_mle", "kl", "ddpo", "ddpp"],
                        default="rw_mle")
    parser.add_argument("--teacher_alpha", type=float, default=1.0)
    parser.add_argument("--reward_norm", choices=["none", "pos", "normal", "top_k"],
                        default="top_k")
    parser.add_argument("--reward_norm_top_k", type=int, default=10,
                        help="K for reward_norm=top_k")
    parser.add_argument("--gkd_lmbda", type=float, default=0.0)
    parser.add_argument("--use_schedule_rollin", action="store_true")
    parser.add_argument("--schedule_max_step", type=int, default=1000)
    parser.add_argument("--old_roll_in", action="store_true", default=True)
    parser.add_argument("--student_roll_in", dest="old_roll_in", action="store_false")
    parser.add_argument("--target_update_interval", type=int, default=20)
    parser.add_argument("--timesteps_per_epoch", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ratio_clip", type=float, default=1e-4)

    # === Evaluation ===
    parser.add_argument("--num_eval_samples", type=int, default=100)

    # === Oracle configs ===
    parser.add_argument("--fa_checkpoint", default="FlashAffinity/checkpoints/value_1.ckpt")
    parser.add_argument("--protein_pdb", default="FlashAffinity/data/protein_test/pdb/2VT4.pdb")
    parser.add_argument("--protein_repr", default="FlashAffinity/data/protein_test/repr/esm3.lmdb")
    parser.add_argument("--protein_id", default="2VT4")
    parser.add_argument("--boltz_cache", default="/home/ariel/.boltz")
    parser.add_argument("--diffusion_samples", type=int, default=4)

    # === DDPP-parity ablation flags (mirror run_active_loop.py / run_ddpp_online.py) ===
    parser.add_argument("--tau_mode", choices=["cvar", "none"], default="none",
                        help="cvar=floor rewards at quantile threshold; none=no clip")
    parser.add_argument("--tau_quantile", type=float, default=0.80,
                        help="quantile (0..1) used when tau_mode=cvar")
    parser.add_argument("--exploration_approach", choices=["none", "A"], default="none",
                        help="A=apply log_r -= gamma * log p_theta(x).detach() in VIDD loss")
    parser.add_argument("--exploration_gamma", type=float, default=0.0,
                        help="coefficient for Approach A; 0 disables")
    parser.add_argument("--invalid_penalty", type=float, default=None,
                        help="If set, invalid SMILES get this fixed reward (wraps oracle).")
    parser.add_argument("--buffer_size", type=int, default=None,
                        help="alias overriding --replay_buffer_size (None = use replay_buffer_size).")

    args = parser.parse_args()

    assert args.max_oracle_calls or args.wall_budget_sec or args.num_steps, \
        "Must specify at least one of --max_oracle_calls, --wall_budget_sec, --num_steps"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # method tag: "vidd_{loss_func}" so each variant is distinguishable in
    # aggregated plots/tables. Example: vidd_rw_mle, vidd_kl, vidd_ddpo, vidd_ddpp.
    method_tag = f"vidd_{args.loss_func}"

    log.info(f"{method_tag}: oracle={args.oracle}, loss_func={args.loss_func}, "
             f"gkd_lmbda={args.gkd_lmbda}, teacher_alpha={args.teacher_alpha}, "
             f"seed={args.seed}")

    # Build oracle + wrap as tensor-returning callable
    base_oracle = build_oracle(args)

    # Validity wrapper: invalid SMILES get a fixed penalty so they propagate
    # into VIDD training as 'low-reward' signals (model is pushed away from
    # invalid token patterns) instead of being silently filtered.
    if args.invalid_penalty is not None:
        from rdkit import Chem as _RDChem
        from rdkit import RDLogger as _RDLogger
        _RDLogger.DisableLog("rdApp.*")
        _inner_oracle = base_oracle
        def _is_valid(smi):
            if not smi or not isinstance(smi, str): return False
            mol = _RDChem.MolFromSmiles(smi)
            if mol is None: return False
            try: _RDChem.SanitizeMol(mol); return True
            except Exception: return False
        def _penalty_wrapped_oracle(smiles_list):
            mask = [_is_valid(s) for s in smiles_list]
            valid_smis = [s for s, v in zip(smiles_list, mask) if v]
            valid_scores = _inner_oracle(valid_smis) if valid_smis else []
            if not isinstance(valid_scores, torch.Tensor):
                valid_scores = torch.tensor(list(valid_scores), dtype=torch.float32)
            it = iter(valid_scores.tolist())
            return torch.tensor(
                [float(next(it)) if v else float(args.invalid_penalty) for v in mask],
                dtype=torch.float32,
            )
        base_oracle = _penalty_wrapped_oracle

    # Per-call logging to oracle_timeline.jsonl
    timeline_path = out_dir / "oracle_timeline.jsonl"
    oracle_state = {"counter": 0, "best": float("-inf"), "t0": None}

    # QED + SA monitors (free)
    try:
        from rdkit import Chem
        from rdkit.Chem import QED as _QED
        def _qed(s):
            m = Chem.MolFromSmiles(s) if s else None
            return round(_QED.qed(m), 4) if m else None
    except ImportError:
        _qed = lambda s: None
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer as _sa
        from rdkit import Chem as _C
        def _sa_score(s):
            m = _C.MolFromSmiles(s) if s else None
            return round(_sa.calculateScore(m), 4) if m else None
    except Exception:
        _sa_score = lambda s: None

    # Track the current "phase" so oracle calls can be distinguished:
    #   - "init"       : initial buffer fill from pretrained
    #   - "{method}_train" : per-step training calls (xs / on-policy x0)
    #   - "{method}_eval"  : final generation evaluation
    oracle_state["phase"] = "init"

    def logged_oracle(smiles_list):
        if oracle_state["t0"] is None:
            oracle_state["t0"] = time.time()
        scores = base_oracle(smiles_list)
        if not isinstance(scores, torch.Tensor):
            scores = torch.tensor(list(scores), dtype=torch.float32)
        scores = scores.cpu().float()

        with open(timeline_path, "a") as tf:
            for smi, sc in zip(smiles_list, scores.tolist()):
                if sc is None or not np.isfinite(sc):
                    continue
                oracle_state["counter"] += 1
                if sc > oracle_state["best"]:
                    oracle_state["best"] = sc
                tf.write(json.dumps({
                    "call": oracle_state["counter"],
                    "smiles": smi,
                    "score": sc,
                    "best_so_far": oracle_state["best"],
                    "method": method_tag,
                    "loss_func": args.loss_func,
                    "phase": oracle_state["phase"],
                    "elapsed_sec": round(time.time() - oracle_state["t0"], 3),
                    "qed": _qed(smi),
                    "sa": _sa_score(smi),
                }) + "\n")
        return scores

    # === Build VIDDTrainer ===
    from genmol.finetune import VIDDTrainer
    trainer = VIDDTrainer(
        model_path=args.model_path,
        reward_fn=logged_oracle,
        lr=args.lr,
        batch_size=args.batch_size,
        replay_buffer_size=(args.buffer_size if args.buffer_size is not None else args.replay_buffer_size),
        ema_decay=args.ema_decay,
        min_add_len=args.min_add_len,
        softmax_temp=args.softmax_temp,
        randomness=args.randomness,
        seed=args.seed,
        verbose=True,
        loss_func=args.loss_func,
        teacher_alpha=args.teacher_alpha,
        reward_norm=args.reward_norm,
        gkd_lmbda=args.gkd_lmbda,
        use_schedule_rollin=args.use_schedule_rollin,
        schedule_max_step=args.schedule_max_step,
        old_roll_in=args.old_roll_in,
        target_update_interval=args.target_update_interval,
        timesteps_per_epoch=args.timesteps_per_epoch,
        grad_clip=args.grad_clip,
        ratio_clip=args.ratio_clip,
    )
    # Plumb DDPP-parity ablation knobs onto the trainer.
    trainer.reward_norm_top_k = args.reward_norm_top_k
    trainer.exploration_approach = args.exploration_approach
    trainer.exploration_gamma = args.exploration_gamma

    # === Pre-seed replay buffer (labeled as "init" in timeline) ===
    # Do this before trainer.train() so the init oracle calls are tagged
    # correctly. trainer.train() will skip its own init fill since the
    # buffer is already non-empty.
    t0 = time.time()
    oracle_state["phase"] = "init"
    n_init = max(trainer.initial_buffer_from_pretrained, trainer.batch_size * 2)
    log.info(f"Seeding replay buffer with {n_init} pretrained samples …")
    trainer._fill_buffer(trainer.pretrained, n_init, label="init-pretrained")
    trainer.cum_oracle_calls = n_init

    # CVaR: set reward_clip_threshold from the init buffer's reward quantile.
    # Mirrors DDPP-LB's tau_mode=cvar but applied once (VIDD trains monolithically;
    # active_loop callers update per-epoch instead).
    if args.tau_mode == "cvar":
        import numpy as _np
        _buf_rewards = [
            float(item["reward"]) for item in trainer.replay_buffer.buffer
            if item.get("reward") is not None and _np.isfinite(item["reward"])
        ]
        if _buf_rewards:
            trainer.reward_clip_threshold = float(_np.quantile(_buf_rewards, args.tau_quantile))
            log.info(f"CVaR floor set to {trainer.reward_clip_threshold:.4f} "
                     f"(quantile {args.tau_quantile} of {len(_buf_rewards)} init samples)")

    # === Training ===
    log.info(f"Starting VIDD training: num_steps={args.num_steps}, "
             f"max_oracle_calls={args.max_oracle_calls}, "
             f"wall_budget={args.wall_budget_sec}")
    oracle_state["phase"] = f"{method_tag}_train"
    trainer.train(
        num_steps=args.num_steps,
        timeout_sec=args.wall_budget_sec,
        max_oracle_calls=args.max_oracle_calls,
    )
    train_wall = time.time() - t0
    log.info(f"Training done: {trainer.global_step} steps, "
             f"{oracle_state['counter']} oracle calls, {train_wall/3600:.2f}hr")

    # === Save checkpoint ===
    trainer.save(str(out_dir / "vidd_checkpoint.pt"))

    # === Final evaluation: generate + oracle-score ===
    log.info(f"Generating {args.num_eval_samples} final samples from EMA model …")
    oracle_state["phase"] = f"{method_tag}_eval"
    final_smiles = trainer.generate(
        num_samples=args.num_eval_samples, use_ema=True,
    )
    final_scores = logged_oracle(final_smiles)

    valid = [(s, float(sc)) for s, sc in zip(final_smiles, final_scores.tolist())
             if s is not None and np.isfinite(sc)]
    valid.sort(key=lambda x: x[1], reverse=True)

    best_score = valid[0][1] if valid else float("-inf")
    top10 = [sc for _, sc in valid[: max(1, len(valid) // 10)]]
    top10_mean = float(np.mean(top10)) if top10 else float("-inf")

    results = {
        "method": method_tag,                     # e.g. "vidd_rw_mle"
        "family": "vidd",                          # coarse grouping
        "loss_func": args.loss_func,               # "rw_mle" | "kl" | "ddpo" | "ddpp"
        "teacher_alpha": args.teacher_alpha,
        "gkd_lmbda": args.gkd_lmbda,
        "reward_norm": args.reward_norm,
        "target_update_interval": args.target_update_interval,
        "timesteps_per_epoch": args.timesteps_per_epoch,
        "oracle": args.oracle,
        "best_score": best_score,
        "top10_mean": top10_mean,
        "best_smiles": valid[0][0] if valid else None,
        "n_valid_final": len(valid),
        "total_oracle_calls": oracle_state["counter"],
        "training_wall_sec": train_wall,
        "total_wall_sec": time.time() - t0,
        "global_step": trainer.global_step,
        "seed": args.seed,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\n{'='*50}")
    log.info(f"Method:         {method_tag}")
    log.info(f"Best score:     {best_score:.4f}")
    log.info(f"Top-10 mean:    {top10_mean:.4f}")
    log.info(f"Oracle calls:   {oracle_state['counter']}")
    log.info(f"Train wall:     {train_wall/3600:.2f}hr")
    log.info(f"Results → {out_dir}/results.json")


if __name__ == "__main__":
    main()
