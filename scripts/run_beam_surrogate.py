"""
Beam search guided by an online-updated surrogate — no generative model fine-tuning.

Pipeline per iteration:
  1. BeamSearchSampler guided by surrogate.predict() (surrogate as forward_op)
     Cold start (epoch 0): uncond generation, random selection.
  2. Score output candidates with FA oracle.
  3. Retrain surrogate on all accumulated (smiles, FA) pairs.
  4. Repeat until wall budget exhausted.

Ablation vs active loop: isolates value of surrogate-guided search
without any DDPP model fine-tuning.
"""
import argparse, json, logging, os, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "FlashAffinity", "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="model_v2.ckpt")
    p.add_argument("--wall_budget_sec", type=float, default=14400)
    p.add_argument("--K", type=int, default=25,    help="candidates per iteration")
    p.add_argument("--beam_width", type=int, default=8)
    p.add_argument("--output_dir", default="outputs/beam_surrogate_4hr")
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    from genmol.samplers import BeamSearchSampler, Sampler
    from genmol.surrogate import SequenceSurrogate

    # ── FA oracle ────────────────────────────────────────────────────────────
    from genmol.rewards.flash_affinity import FlashAffinityForwardOp
    fa_model = FlashAffinityForwardOp(
        protein_pdb="FlashAffinity/data/protein_test/pdb/2VT4.pdb",
        protein_repr_path="FlashAffinity/data/protein_test/repr/esm3.lmdb",
        protein_id="2VT4",
        checkpoint_paths=["FlashAffinity/checkpoints/value_1.ckpt"],
        task="value",
    )
    def fa_oracle(smiles_list):
        t = fa_model(smiles_list)
        return [float(v) if v != 0.0 else None for v in t.tolist()]

    # ── Surrogate ────────────────────────────────────────────────────────────
    surrogate = SequenceSurrogate()

    # ── Main loop ────────────────────────────────────────────────────────────
    all_pairs = []   # (smiles, fa_score)
    best = float("-inf")
    t_start = time.time()
    epoch = 0

    while time.time() - t_start < args.wall_budget_sec:
        t0 = time.time()
        surrogate_ready = surrogate.is_fitted and len(all_pairs) >= 4

        # ── Step 1: generate candidates ─────────────────────────────────────
        if surrogate_ready:
            def _surrogate_forward_op(smiles_list):
                valid_mask = [s is not None and isinstance(s, str) and len(s) > 0
                              for s in smiles_list]
                valid_smiles = [s for s, ok in zip(smiles_list, valid_mask) if ok]
                out = torch.full((len(smiles_list),), -1e9)
                if not valid_smiles:
                    return out
                scores = surrogate.predict(valid_smiles)
                valid_idx = [i for i, ok in enumerate(valid_mask) if ok]
                out[valid_idx] = torch.tensor(scores, dtype=torch.float)
                return out

            sampler = BeamSearchSampler(
                args.model_path,
                forward_op=_surrogate_forward_op,
                beam_width=args.beam_width,
                
            )
            candidates = sampler.de_novo_generation(
                num_samples=args.K,
                softmax_temp=0.8,
                randomness=0.5,
                min_add_len=40,
            )
            log.info(f"[epoch {epoch}] Beam search (surrogate) → {len(candidates)} candidates")
        else:
            # Cold start: uncond generation
            base_sampler = Sampler(args.model_path, device=str(device))
            candidates = base_sampler.de_novo_generation(
                num_samples=args.K,
                softmax_temp=0.8,
                randomness=0.5,
                min_add_len=40,
            )
            candidates = [c for c in candidates if c is not None and isinstance(c, str) and len(c) > 0]
            log.info(f"[epoch {epoch}] Cold start: {len(candidates)} uncond candidates")

        if not candidates:
            epoch += 1
            continue

        # ── Step 2: FA oracle ────────────────────────────────────────────────
        fa_scores = fa_oracle(candidates)
        new_pairs = [(s, sc) for s, sc in zip(candidates, fa_scores)
                     if sc is not None and not np.isnan(sc)]
        all_pairs.extend(new_pairs)
        if new_pairs:
            best = max(best, max(sc for _, sc in new_pairs))

        # ── Step 3: retrain surrogate ────────────────────────────────────────
        if len(all_pairs) >= 8:
            smiles_fit = [s for s, _ in all_pairs]
            scores_fit = [sc for _, sc in all_pairs]
            surrogate.fit(smiles_fit, scores_fit)

        elapsed = time.time() - t0
        log.info(
            f"[epoch {epoch}] f*={best:.4f} | oracle_calls={len(all_pairs)} | t={elapsed:.1f}s"
        )
        epoch += 1

    # ── Save results ─────────────────────────────────────────────────────────
    top_pairs = sorted(all_pairs, key=lambda x: x[1], reverse=True)
    top10_mean = float(np.mean([sc for _, sc in top_pairs[:10]])) if top_pairs else float("nan")
    results = {
        "method": "beam_surrogate_no_finetune",
        "best_fa": best,
        "top10_mean": top10_mean,
        "oracle_calls": len(all_pairs),
        "epochs": epoch,
        "top_smiles": [s for s, _ in top_pairs[:20]],
        "top_scores": [sc for _, sc in top_pairs[:20]],
    }
    out_path = Path(args.output_dir) / "results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Final: f*={best:.4f} | top10={top10_mean:.4f} | oracle_calls={len(all_pairs)}")
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
