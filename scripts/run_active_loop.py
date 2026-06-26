"""
Entry point for the adaptive active learning loop.

Usage:
    python scripts/run_active_loop.py \
        --model_path model_v2.ckpt \
        --n_epochs 20 \
        --M 500 \
        --K 25 \
        --tau_mode cvar \
        --output_dir outputs/active_loop_cvar

    # best-so-far variant:
    python scripts/run_active_loop.py \
        --model_path model_v2.ckpt \
        --tau_mode best_so_far
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--wall_budget_sec", type=float, default=None,
                        help="If set, run for this many seconds instead of n_epochs (e.g. 14400 = 4hr)")
    parser.add_argument("--M", type=int, default=500)
    parser.add_argument("--K", type=int, default=25)
    parser.add_argument("--tau_mode", choices=["cvar", "best_so_far", "none", "anneal"], default="cvar")
    parser.add_argument("--tau_quantile", type=float, default=0.80)
    parser.add_argument("--reward_shift", type=float, default=3.0)
    parser.add_argument("--ddpp_steps", type=int, default=50)
    parser.add_argument("--no_ensemble", action="store_true",
                        help="Disable Thompson Sampling (DDPP+CVaR ablation, no ensemble)")
    parser.add_argument("--selection",
                        choices=["thompson", "beam", "thompson_beam", "thompson_mcts",
                                 "bandit_ensemble", "bandit_global", "bandit_ts", "bandit_ts_upper"],
                        default="thompson")
    parser.add_argument("--beam_width", type=int, default=4)
    parser.add_argument("--mcts_rollout_budget", type=int, default=None)
    parser.add_argument("--bandit_z", type=float, default=1.96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_ensemble", type=int, default=10)
    parser.add_argument("--ensemble_epochs", type=int, default=100)
    parser.add_argument("--output_dir", default="outputs/active_loop")
    parser.add_argument("--protein_pdb", default="FlashAffinity/data/protein_test/pdb/2VT4.pdb")
    parser.add_argument("--protein_repr", default="FlashAffinity/data/protein_test/repr/esm3.lmdb")
    parser.add_argument("--protein_id", default="2VT4")
    parser.add_argument("--fa_checkpoint", default="FlashAffinity/checkpoints/value_1.ckpt")
    parser.add_argument("--oracle", default="fa",
                        help="Oracle name: fa | boltz | any name registered via register_reward().")
    parser.add_argument("--oracle_module", default=None,
                        help="Python import path of a module that calls register_reward(), "
                             "e.g. examples.custom_oracle or mypackage.my_oracle")
    parser.add_argument("--boltz_cache", default=os.path.expanduser("~/.boltz"))
    parser.add_argument("--diffusion_samples", type=int, default=4, help="Boltz diffusion samples per inference")
    parser.add_argument("--gpu", type=int, default=0)
    # DDPP hyperparams
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_logz", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--warmup_logz_steps", type=int, default=100)
    parser.add_argument("--exploration_approach", choices=["none", "A", "B", "C"], default="none")
    parser.add_argument("--exploration_gamma", type=float, default=0.0)
    parser.add_argument("--negscore_estimator", choices=["rough", "llada"],
                        default="rough",
                        help="Estimator for log p_θ(x) inside Approach A/B/C neg-score. "
                             "\"rough\"=single sample (biased, high var); "
                             "\"llada\"=Alg-3 with L/l weight, n_mc samples (unbiased).")
    parser.add_argument("--negscore_n_mc", type=int, default=1,
                        help="MC samples for negscore_estimator=llada (ignored for rough).")
    parser.add_argument("--kl_lambda", type=float, default=0.0)
    parser.add_argument("--use_mara", action="store_true")
    parser.add_argument("--invalid_penalty", type=float, default=None,
                        help="If set, invalid SMILES get this fixed reward (else NaN-filtered as before)")
    parser.add_argument("--tau_anneal_start", type=float, default=0.30)
    parser.add_argument("--tau_anneal_end", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=None,
                        help="replay buffer capacity (None = use default ~10000; 0 = batch-only no replay)")
    parser.add_argument("--trainer", choices=["ddpp", "vidd"], default="ddpp",
                        help="base finetune trainer wrapped by Thompson + ensemble")
    parser.add_argument("--vidd_loss_func", choices=["kl", "rw_mle", "ddpo", "ddpp"], default="kl",
                        help="VIDD loss variant (only used when --trainer vidd)")
    parser.add_argument("--vidd_teacher_alpha", type=float, default=1.0)
    parser.add_argument("--vidd_gkd_lmbda", type=float, default=0.5,
                        help="VIDD on-policy roll-in fraction; 0=pure pretrain roll-in")
    parser.add_argument("--vidd_reward_norm", choices=["none", "pos", "normal", "top_k"],
                        default="top_k", help="VIDD reward normalization (default top_k)")
    parser.add_argument("--vidd_reward_norm_top_k", type=int, default=10,
                        help="K for vidd_reward_norm=top_k")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import random, numpy as np, torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from genmol.finetune.ddpp import DDPPLBTrainer
    from genmol.active_loop import ActiveLoopConfig, run_active_loop

    # ── Custom oracle (loaded before built-ins so it can override names) ───
    if args.oracle_module:
        import importlib
        importlib.import_module(args.oracle_module)

    # ── FA oracle ──────────────────────────────────────────────────────────
    _root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, _root)
    sys.path.insert(0, os.path.join(_root, "src"))
    sys.path.insert(0, os.path.join(_root, "FlashAffinity", "src"))

    if args.oracle == "boltz":
        from genmol.rewards.boltz import BoltzAffinityReward
        _boltz_model = BoltzAffinityReward(cache_dir=args.boltz_cache, gpu_id=args.gpu,
                                            diffusion_samples=args.diffusion_samples)
        def fa_oracle(smiles_list: list[str]) -> list[float]:
            try:
                scores = _boltz_model(smiles_list)
                if hasattr(scores, "tolist"):
                    scores = scores.tolist()
                return [float(s) if s is not None else None for s in scores]
            except Exception as e:
                print(f"[boltz oracle] batch failed: {e}; returning Nones", flush=True)
                return [None] * len(smiles_list)
    else:
        oracle_params = {
            "protein_pdb": args.protein_pdb,
            "protein_repr_path": args.protein_repr,
            "protein_id": args.protein_id,
            "checkpoint_paths": [args.fa_checkpoint],
            "task": "value",
        }
        try:
            from genmol.rewards.flash_affinity import FlashAffinityForwardOp
            _fa_model = FlashAffinityForwardOp(**oracle_params)
            def fa_oracle(smiles_list: list[str]) -> list[float]:
                t = _fa_model(smiles_list)
                return [float(v) if v != 0.0 else None for v in t.tolist()]
        except Exception:
            from evals.flash_affinity import run_flash_affinity
            def fa_oracle(smiles_list: list[str]) -> list[float]:
                return run_flash_affinity(smiles_list, **oracle_params)

    # ── DDPP trainer ───────────────────────────────────────────────────────
    if args.trainer == "ddpp":
        _buf_kwargs = {}
        if args.buffer_size is not None:
            _buf_kwargs["buffer_size"] = max(args.batch_size, args.buffer_size)
        trainer = DDPPLBTrainer(
            model_path=args.model_path,
            reward_fn=None,          # active loop manages oracle calls directly
            beta=args.beta,
            lr=args.lr,
            lr_logz=args.lr_logz,
            batch_size=args.batch_size,
            warmup_logz_steps=args.warmup_logz_steps,
            refill_interval=0,       # no auto-refill; active loop controls generation
            **_buf_kwargs,
        )
        trainer.kl_lambda = args.kl_lambda
        trainer.use_mara = args.use_mara
    elif args.trainer == "vidd":
        from genmol.finetune import VIDDTrainer
        _buf_kw_vidd = {}
        if args.buffer_size is not None:
            _buf_kw_vidd["replay_buffer_size"] = max(args.batch_size, args.buffer_size)

        # VIDD's kl/ddpo/ddpp loss paths score on-policy x_s inside train_step.
        # Late-bound closure: at training time fa_oracle will be the validity-
        # wrapped version (if --invalid_penalty was set).
        def _vidd_oracle(smiles_list):
            return fa_oracle(smiles_list)

        trainer = VIDDTrainer(
            model_path=args.model_path,
            reward_fn=_vidd_oracle,
            lr=args.lr,
            batch_size=args.batch_size,
            refill_interval=0,
            seed=args.seed,
            loss_func=args.vidd_loss_func,
            teacher_alpha=args.vidd_teacher_alpha,
            gkd_lmbda=args.vidd_gkd_lmbda,
            reward_norm=args.vidd_reward_norm,
            **_buf_kw_vidd,
        )
    else:
        raise ValueError(f"Unknown --trainer {args.trainer!r}")
    # VIDD-only: reward_norm top_k value (no-op for DDPP path).
    if args.trainer == "vidd":
        trainer.reward_norm_top_k = args.vidd_reward_norm_top_k
    # Plumb shared reward-shaping attributes (both trainers honor these).
    trainer.exploration_approach = args.exploration_approach
    trainer.exploration_gamma = args.exploration_gamma
    # Neg-score (Approach A/B/C) log p_θ estimator: rough vs LLaDA Alg-3.
    if hasattr(trainer, "negscore_estimator"):
        trainer.negscore_estimator = args.negscore_estimator
        trainer.negscore_n_mc = args.negscore_n_mc

    # Validity penalty wrapper: invalid SMILES get a fixed penalty (signal flows
    # into DDPP-LB as 'low reward', model is pushed away from invalid patterns).
    if args.invalid_penalty is not None:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
        # Tokenizer-aware validity: SMILES that pass RDKit but produce
        # out-of-vocab token ids crash the embedding layer with a CUDA assert
        # (e.g., exotic [PH]3 phosphorus). Reject them here so they get the
        # invalid_penalty and never reach the model.
        _tk = trainer.tokenizer
        _vsz = getattr(trainer.finetuned.config.model, "vocab_size", None) or _tk.vocab_size
        _max_len = trainer.finetuned.config.model.max_position_embeddings
        def _is_valid(smi):
            if not smi or not isinstance(smi, str): return False
            mol = Chem.MolFromSmiles(smi)
            if mol is None: return False
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return False
            try:
                enc = _tk([smi], return_tensors="pt", truncation=True, max_length=_max_len)
                ids = enc["input_ids"]
                if ids.numel() == 0 or ids.max().item() >= _vsz or ids.min().item() < 0:
                    return False
            except Exception:
                return False
            return True
        _inner_oracle = fa_oracle
        def fa_oracle(smiles_list):
            valid_mask = [_is_valid(s) for s in smiles_list]
            valid_smis = [s for s, v in zip(smiles_list, valid_mask) if v]
            try:
                valid_scores = _inner_oracle(valid_smis) if valid_smis else []
            except Exception:
                # Batch crashed (e.g., FA conformer-gen exception) — fall back per-SMILES
                valid_scores = []
                for _s in valid_smis:
                    try:
                        v = _inner_oracle([_s])
                        valid_scores.append(v[0] if v else None)
                    except Exception:
                        valid_scores.append(None)
            it = iter(valid_scores)
            out = []
            for v in valid_mask:
                if not v:
                    out.append(float(args.invalid_penalty))
                else:
                    sc = next(it, None)
                    out.append(float(sc) if sc is not None else float(args.invalid_penalty))
            return out


    # ── Config ─────────────────────────────────────────────────────────────
    cfg = ActiveLoopConfig(
        n_epochs=args.n_epochs,
        wall_budget_sec=args.wall_budget_sec,
        M=args.M,
        K=args.K,
        reward_shift=args.reward_shift,
        tau_mode=args.tau_mode,
        tau_quantile=args.tau_quantile,
        use_ensemble=not args.no_ensemble,
        selection=args.selection,
        beam_width=args.beam_width,
        mcts_rollout_budget=args.mcts_rollout_budget,
        bandit_z=args.bandit_z,
        ddpp_steps_per_epoch=args.ddpp_steps,
        n_ensemble=args.n_ensemble,
        ensemble_epochs=args.ensemble_epochs,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # ── Run ────────────────────────────────────────────────────────────────
    results = run_active_loop(trainer, fa_oracle, cfg)
    print(f"\nBest molecule: {results['best_smiles']}")
    print(f"Best FA score: {results['f_star']:.4f}")
    print(f"Total oracle calls: {len(results['all_pairs'])}")


if __name__ == "__main__":
    main()
