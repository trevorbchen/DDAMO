"""
DFKC (Discrete Feynman-Kac Corrector) with FA oracle.
Runs for a wall-clock budget, logging every oracle call.

Usage:
    python scripts/run_dfkc_fa.py \
        --model_path model_v2.ckpt \
        --wall_budget_sec 7200 \
        --gpu 0 --seed 0 \
        --output_dir outputs/dfkc_fa
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
    parser.add_argument("--wall_budget_sec", type=float, default=7200)
    parser.add_argument("--max_oracle_calls", type=int, default=None,
                        help="early-exit after this many oracle (FA) calls")
    parser.add_argument("--num_particles", type=int, default=16)
    parser.add_argument("--beta", type=float, default=15.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scoring_mode", default="completion", choices=["completion", "tweedie", "current"])
    parser.add_argument("--output_dir", default="outputs/dfkc_fa")
    parser.add_argument("--protein_pdb", default="FlashAffinity/data/protein_test/pdb/2VT4.pdb")
    parser.add_argument("--protein_repr", default="FlashAffinity/data/protein_test/repr/esm3.lmdb")
    parser.add_argument("--protein_id", default="2VT4")
    parser.add_argument("--fa_checkpoint", default="FlashAffinity/checkpoints/value_1.ckpt")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # FA oracle
    from genmol.rewards.flash_affinity import FlashAffinityForwardOp
    _fa_model = FlashAffinityForwardOp(
        protein_pdb=args.protein_pdb,
        protein_repr_path=args.protein_repr,
        protein_id=args.protein_id,
        checkpoint_paths=[args.fa_checkpoint],
        task="value",
    )

    # Oracle timeline logging
    os.makedirs(args.output_dir, exist_ok=True)
    timeline_path = os.path.join(args.output_dir, "oracle_timeline.jsonl")
    call_counter = [0]
    best_so_far = [float("-inf")]
    wall_start = time.time()

    # QED + SA monitors
    try:
        from rdkit import Chem
        from rdkit.Chem import QED as _QED
        def _qed(smi):
            mol = Chem.MolFromSmiles(smi)
            return round(_QED.qed(mol), 4) if mol else None
    except ImportError:
        _qed = lambda s: None
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer as _sa
        from rdkit import Chem as _C2
        def _sa_score(smi):
            mol = _C2.MolFromSmiles(smi)
            return round(_sa.calculateScore(mol), 4) if mol else None
    except Exception:
        _sa_score = lambda s: None

    def logged_fa(smiles_list):
        scores = _fa_model(smiles_list)
        with open(timeline_path, "a") as tf:
            for smi, sc in zip(smiles_list, scores.tolist()):
                if sc == 0.0:
                    continue
                call_counter[0] += 1
                best_so_far[0] = max(best_so_far[0], sc)
                tf.write(json.dumps({
                    "call": call_counter[0],
                    "smiles": smi,
                    "score": sc,
                    "best_so_far": best_so_far[0],
                    "elapsed_sec": round(time.time() - wall_start, 3),
                    "qed": _qed(smi),
                    "sa": _sa_score(smi),
                }) + "\n")
        return scores

    # Load model and create DFKC sampler
    from genmol.samplers import DFKCSampler, load_model_from_path
    from genmol.rewards import KLPenalizedReward
    base_model = load_model_from_path(args.model_path)
    kl_reward = KLPenalizedReward(logged_fa, base_model, lam=0.01)

    sampler = DFKCSampler(
        model=base_model, forward_op=kl_reward,
        num_particles=args.num_particles, beta=args.beta, mode="reward",
        beta_schedule="sqrt", score_start=0.7,
        score_interval=5, score_interval_late=2,
        late_phase=0.8, ess_threshold=0.3,
        rollout=False, scoring_mode=args.scoring_mode, elite_ratio=0.5,
        elite_buffer_size=50, elite_mask_ratio=0.5,
    )

    log.info(f"DFKC FA: particles={args.num_particles}, beta={args.beta}, "
             f"budget={args.wall_budget_sec/3600:.1f}hr, seed={args.seed}")

    # Run for wall OR oracle budget (whichever hits first)
    t0 = time.time()
    while time.time() - t0 < args.wall_budget_sec:
        if args.max_oracle_calls is not None and call_counter[0] >= args.max_oracle_calls:
            log.info(f"Oracle budget exhausted: {call_counter[0]} >= {args.max_oracle_calls}")
            break
        # Pass remaining oracle budget to the sampler so it can stop mid-round.
        rem = (args.max_oracle_calls - call_counter[0]) if args.max_oracle_calls else None
        smiles = sampler.de_novo_generation(
            num_samples=1, softmax_temp=1.0, randomness=0.3, min_add_len=60,
            max_reward_evals=rem,
        )
        if call_counter[0] % 100 == 0:
            log.info(f"calls={call_counter[0]}, best={best_so_far[0]:.4f}, "
                     f"elapsed={time.time()-t0:.0f}s")

    log.info(f"Done: {call_counter[0]} calls, best_fa={best_so_far[0]:.4f}")

    # Save results
    results = {
        "method": "dfkc_fa",
        "best_fa": best_so_far[0],
        "total_oracle_calls": call_counter[0],
        "wall_sec": time.time() - wall_start,
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Best molecule: {best_so_far[0]:.4f}")


if __name__ == "__main__":
    main()
