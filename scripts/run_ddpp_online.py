"""
DDPP Online: generate → score ALL with oracle → CVaR fine-tune → repeat.
No surrogate, no selection. Pure online finetuning baseline.

Usage:
    python scripts/run_ddpp_online.py --model_path model_v2.ckpt --wall_budget_sec 7200 --gpu 0 --seed 0
    python scripts/run_ddpp_online.py --model_path model_v2.ckpt --max_calls 800 --gpu 0 --seed 0  # call-budgeted
"""
import argparse, json, logging, os, sys, time, random
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

_root = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "FlashAffinity", "src"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--wall_budget_sec", type=float, default=None,
                        help="Wall-clock budget (e.g. 7200 for 2hr)")
    parser.add_argument("--max_calls", type=int, default=None,
                        help="Oracle call budget (e.g. 800 for Boltz)")
    parser.add_argument("--batch_size", type=int, default=25,
                        help="Molecules generated + scored per epoch")
    parser.add_argument("--tau_mode", choices=["cvar", "none", "anneal"], default="cvar")
    parser.add_argument("--tau_quantile", type=float, default=0.80)
    parser.add_argument("--tau_anneal_start", type=float, default=0.30, help="starting quantile for anneal mode (permissive)")
    parser.add_argument("--tau_anneal_end", type=float, default=0.99, help="ending quantile for anneal mode (strict)")
    parser.add_argument("--tau_anneal_plateau_at", type=float, default=1.0, help="wall fraction at which to reach end quantile and plateau (1.0=no plateau, ramp full wall)")
    parser.add_argument("--fresh_fraction", type=float, default=0.0, help="fraction of each batch drawn from this-epoch samples (0=off)")
    parser.add_argument("--eviction", choices=["fifo", "priority"], default="fifo", help="buffer eviction policy")
    parser.add_argument("--reward_floor", type=float, default=1e-8, help="floor for (reward-threshold).clamp; 1.0 = log_r=0 neutral, 1e-8 = strong push-down (default)")
    parser.add_argument("--exploration_approach", choices=["none", "A", "B", "C"], default="none", help="Exploration-bonus reward modification")
    parser.add_argument("--exploration_gamma", type=float, default=0.0, help="γ (approach A/B) or μ (approach C) coefficient; default 1.0 when approach != none")
    parser.add_argument("--negscore_estimator", choices=["rough", "llada"], default="rough",
                        help="log p_θ estimator for neg-score: rough=single-sample (current), llada=Alg-3 n_mc-averaged with L/l weight")
    parser.add_argument("--negscore_n_mc", type=int, default=1,
                        help="MC samples for negscore_estimator=llada (ignored for rough)")
    parser.add_argument("--kl_lambda", type=float, default=0.0, help="λ coefficient for KL regularization term (approach C only)")
    parser.add_argument("--use_mara", action="store_true", help="Enable MARA Mode Anchored Reward Augmentation (GX-Chen 2025)")
    parser.add_argument("--refill_interval", type=int, default=-1, help="DDPP buffer auto-refill cadence; -1 = auto (0 if buffer<=1, else default 50)")
    parser.add_argument("--invalid_penalty", type=float, default=None, help="If set, invalid SMILES get this fixed reward (else NaN-filtered as before)")
    parser.add_argument("--ddpp_steps", type=int, default=50)
    parser.add_argument("--reward_shift", type=float, default=3.0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    # DDPP params
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lr_logz", type=float, default=5e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--warmup_logz_steps", type=int, default=100)
    parser.add_argument("--buffer_size", type=int, default=10000,
                        help="Replay buffer size (0 = no replay, only current epoch data)")
    # Oracle type
    parser.add_argument("--oracle", choices=["fa", "boltz"], default="fa")
    parser.add_argument("--oracle_module", default=None,
                        help="Python import path of module that registers custom oracles")
    parser.add_argument("--fa_checkpoint", default="FlashAffinity/checkpoints/value_1.ckpt")
    parser.add_argument("--boltz_cache", default=os.path.expanduser("~/.boltz"))
    parser.add_argument("--diffusion_samples", type=int, default=4)
    args = parser.parse_args()

    assert args.wall_budget_sec or args.max_calls, "Must specify --wall_budget_sec or --max_calls"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    timeline_path = os.path.join(args.output_dir, "oracle_timeline.jsonl")
    log_path = os.path.join(args.output_dir, "active_loop_log.jsonl")

    # QED + SA
    try:
        from rdkit import Chem; from rdkit.Chem import QED as _QED
        def _qed(s):
            m = Chem.MolFromSmiles(s)
            return round(_QED.qed(m), 4) if m else None
    except ImportError:
        _qed = lambda s: None
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer as _sa; from rdkit import Chem as _C
        def _sa_score(s):
            m = _C.MolFromSmiles(s)
            return round(_sa.calculateScore(m), 4) if m else None
    except Exception:
        _sa_score = lambda s: None

    # Oracle
    if args.oracle == "boltz":
        from genmol.rewards.boltz import BoltzAffinityReward
        _oracle = BoltzAffinityReward(cache_dir=args.boltz_cache, gpu_id=args.gpu,
                                       diffusion_samples=args.diffusion_samples)
        def oracle_fn(smiles_list):
            scores = _oracle(smiles_list)
            return [float(s) for s in scores.tolist()]
    else:
        from genmol.rewards.flash_affinity import FlashAffinityForwardOp
        _fa = FlashAffinityForwardOp(
            protein_pdb="FlashAffinity/data/protein_test/pdb/2VT4.pdb",
            protein_repr_path="FlashAffinity/data/protein_test/repr/esm3.lmdb",
            protein_id="2VT4", checkpoint_paths=[args.fa_checkpoint], task="value")
        def oracle_fn(smiles_list):
            t = _fa(smiles_list)
            return [float(v) if v != 0.0 else -5.0 for v in t.tolist()]

    # Validity penalty wrapper: invalid SMILES get a fixed penalty score so they
    # propagate into DDPP-LB training as 'low-reward' signals (model is pushed
    # away from invalid token patterns) instead of silently filtered.
    if args.invalid_penalty is not None:
        from rdkit import Chem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")  # suppress RDKit warnings
        def _is_valid(smi):
            if not smi or not isinstance(smi, str):
                return False
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return False
            try:
                Chem.SanitizeMol(mol)
                return True
            except Exception:
                return False
        _inner_oracle = oracle_fn
        def oracle_fn(smiles_list):
            valid_mask = [_is_valid(s) for s in smiles_list]
            valid_smis = [s for s, v in zip(smiles_list, valid_mask) if v]
            valid_scores = _inner_oracle(valid_smis) if valid_smis else []
            it = iter(valid_scores)
            return [float(next(it)) if v else float(args.invalid_penalty) for v in valid_mask]

    # DDPP trainer
    from genmol.finetune.ddpp import DDPPLBTrainer
    # buffer_size semantics:
    #   N>=1: explicit capacity
    #   0:    online (= batch_size, one mini-batch worth, no replay history)
    if args.buffer_size <= 0:
        _buf = 16  # = trainer.batch_size default; one mini-batch fully retained
    else:
        _buf = args.buffer_size
    trainer = DDPPLBTrainer(
        model_path=os.path.realpath(args.model_path),
        reward_fn=oracle_fn, beta=args.beta, lr=args.lr, lr_logz=args.lr_logz,
        batch_size=16, warmup_logz_steps=args.warmup_logz_steps,
        buffer_size=_buf)
    trainer.fresh_fraction = args.fresh_fraction
    # Recreate buffer with chosen eviction policy
    from genmol.finetune.ddpp import ReplayBuffer
    trainer.buffer = ReplayBuffer(capacity=_buf, pad_token_id=trainer.tokenizer.pad_token_id, eviction=args.eviction)
    trainer.reward_floor = args.reward_floor
    trainer.exploration_approach = args.exploration_approach
    trainer.exploration_gamma = args.exploration_gamma if args.exploration_gamma > 0 else (1.0 if args.exploration_approach != "none" else 0.0)
    # Neg-score (Approach A/B/C) log p_θ estimator: rough vs LLaDA Alg-3.
    trainer.negscore_estimator = args.negscore_estimator
    trainer.negscore_n_mc = args.negscore_n_mc
    trainer.kl_lambda = args.kl_lambda
    trainer.use_mara = args.use_mara
    # Auto-disable refill for nobuf runs (buffer too small for refill loop, causes silent Boltz oracle waste)
    if args.refill_interval >= 0:
        trainer.refill_interval = args.refill_interval
    elif _buf <= 1:
        trainer.refill_interval = 0  # disable: nothing to refill into a 1-slot buffer

    wall_start = time.time()
    call_counter = 0
    best_so_far = float("-inf")
    all_pairs = []  # (smiles, score)
    epoch = 0
    cumulative_diffusion_passes = 0
    cumulative_ddpp_steps = 0

    log.info(f"DDPP Online: tau_mode={args.tau_mode}, batch={args.batch_size}, "
             f"oracle={args.oracle}, seed={args.seed}")

    def budget_left():
        if args.max_calls and call_counter >= args.max_calls:
            return False
        if args.wall_budget_sec and (time.time() - wall_start) >= args.wall_budget_sec:
            return False
        return True

    while budget_left():
        t0 = time.time()

        # Generate
        t_gen_start = time.time()
        batch = trainer.generate(args.batch_size,
                                  softmax_temp=0.8, randomness=0.3, min_add_len=60)
        raw_batch = list(getattr(trainer, "_last_raw_smiles", batch))

        n_attempted = getattr(trainer, '_last_n_attempted', args.batch_size)
        n_valid = getattr(trainer, '_last_n_valid', len(batch))
        cumulative_diffusion_passes += n_attempted
        t_gen = time.time() - t_gen_start

        if not batch:
            epoch += 1
            continue

        # Score ALL with oracle
        t_oracle_start = time.time()
        scores = oracle_fn(batch)
        t_oracle = time.time() - t_oracle_start

        # Log each call
        new_pairs = []
        with open(timeline_path, "a") as tf:
            for smi, sc in zip(batch, scores):
                if sc is None or (isinstance(sc, float) and sc != sc):
                    continue
                # Clamp -inf and extreme negatives to -10 to prevent DDPP gradient explosion
                if isinstance(sc, float) and sc < -10.0:
                    sc = -10.0
                call_counter += 1
                best_so_far = max(best_so_far, sc)
                tf.write(json.dumps({"call": call_counter, "smiles": smi, "score": sc,
                    "best_so_far": best_so_far, "epoch": epoch,
                    "elapsed_sec": round(time.time() - wall_start, 3),
                    "qed": _qed(smi), "sa": _sa_score(smi)}) + "\n")
                new_pairs.append((smi, sc))
        all_pairs.extend(new_pairs)

        # CVaR fine-tune
        t_ddpp_start = time.time()
        if all_pairs:
            raw_scores = [s for _, s in all_pairs]
            if args.tau_mode == "cvar":
                tau = float(np.quantile(raw_scores, args.tau_quantile))
            elif args.tau_mode == "anneal":
                wall_frac = min((time.time() - wall_start) / max(args.wall_budget_sec or 1e9, 1.0), 1.0)
                _ramp = (wall_frac / args.tau_anneal_plateau_at) if args.tau_anneal_plateau_at > 0 else 1.0
                _ramp = min(_ramp, 1.0)
                tau_q = args.tau_anneal_start + (args.tau_anneal_end - args.tau_anneal_start) * _ramp
                tau = float(np.quantile(raw_scores, tau_q))
            else:
                tau = min(raw_scores) - 1.0

            trainer.reward_clip_threshold = tau + args.reward_shift
            shifted = [(smi, sc + args.reward_shift) for smi, sc in new_pairs]
            trainer.add_scored_molecules([s for s, _ in shifted], [r for _, r in shifted])
            trainer.train(max_steps=args.ddpp_steps)
            cumulative_ddpp_steps += args.ddpp_steps
        t_ddpp = time.time() - t_ddpp_start

        elapsed = time.time() - t0

        # Log epoch
        epoch_log = {
            "epoch": epoch,
            "f_star": best_so_far,
            "n_oracle_calls": call_counter,
            "epoch_fa_scores": [sc for _, sc in new_pairs],
            "epoch_fa_smiles": [smi for smi, _ in new_pairs],
            "epoch_all_smiles": [
                {"smiles": _s, "valid": _s is not None,
                 "selected": (_s is not None and _s in set(ss for ss, _ in new_pairs))}
                for _s in (raw_batch if "raw_batch" in dir() else [])
            ],
            "t_gen": round(t_gen, 3),
            "t_surrogate": 0.0,
            "t_oracle": round(t_oracle, 3),
            "t_ddpp": round(t_ddpp, 3),
            "t_ensemble": 0.0,
            "n_attempted": n_attempted,
            "n_valid": n_valid,
            "cumulative_diffusion_passes": cumulative_diffusion_passes,
            "cumulative_ddpp_steps": cumulative_ddpp_steps,
            "wall_sec": round(time.time() - wall_start, 3),
            "elapsed_sec": round(elapsed, 3),
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(epoch_log) + "\n")

        if epoch % 5 == 0:
            log.info(f"[epoch {epoch}] f*={best_so_far:.4f} | calls={call_counter} | "
                     f"t={elapsed:.1f}s | wall={time.time()-wall_start:.0f}s")
        epoch += 1

    total_wall = time.time() - wall_start
    log.info(f"Done: {call_counter} calls, best={best_so_far:.4f}, wall={total_wall/3600:.2f}hr")

    top10 = sorted([s for _, s in all_pairs], reverse=True)[:10]
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({
            "method": "ddpp_online_" + ({"cvar":"cvar","none":"nocvar","anneal":"anneal"}[args.tau_mode]),
            "oracle": args.oracle,
            "best_score": best_so_far,
            "top10_mean": float(np.mean(top10)) if top10 else None,
            "total_calls": call_counter,
            "wall_sec": total_wall,
            "seed": args.seed,
            "tau_mode": args.tau_mode,
        }, f, indent=2)

if __name__ == "__main__":
    main()