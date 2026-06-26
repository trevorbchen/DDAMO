# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import warnings
warnings.filterwarnings('ignore')

import itertools
import pickle
import torch
import random
import safe as sf
from rdkit import Chem
from genmol.utils.utils_chem import safe_to_smiles, filter_by_substructure, mix_sequences, Slicer
from genmol.utils.bracket_safe_converter import BracketSAFEConverter, bracketsafe2safe
from genmol.model import GenMol


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))


def _tanimoto_diversity(smiles_list, max_pairs=5000):
    """Average pairwise Tanimoto distance for a list of SMILES.

    Returns a float in [0, 1] where 1 = maximally diverse.
    Subsamples pairs if the list is large to keep runtime bounded.
    """
    from rdkit.Chem import AllChem, DataStructs
    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    if len(fps) < 2:
        return 0.0
    n = len(fps)
    if n * (n - 1) // 2 <= max_pairs:
        total_sim = 0.0
        count = 0
        for i in range(n):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            total_sim += sum(sims)
            count += len(sims)
    else:
        import random as _rng
        total_sim = 0.0
        count = max_pairs
        for _ in range(max_pairs):
            i, j = _rng.sample(range(n), 2)
            total_sim += DataStructs.TanimotoSimilarity(fps[i], fps[j])
    return 1.0 - total_sim / count if count > 0 else 0.0


def load_model_from_path(path):
    model = GenMol.load_from_checkpoint(path)
    model.backbone.eval()
    if model.ema:
        model.ema.store(itertools.chain(model.backbone.parameters()))
        model.ema.copy_to(itertools.chain(model.backbone.parameters()))
    return model


def decode_smiles(model, x, fix=True):
    """Decode token tensor → list of SMILES (None for invalid).

    Pipeline: token ids → SAFE string → canonical SMILES.
    Keeps only the largest dot-separated fragment.
    """
    strings = model.tokenizer.batch_decode(x, skip_special_tokens=True)
    out = []
    for s in strings:
        try:
            if model.config.training.get("use_bracket_safe"):
                smi = safe_to_smiles(bracketsafe2safe(s), fix=fix)
            else:
                smi = safe_to_smiles(s, fix=fix)
        except Exception:
            smi = None
        if smi:
            smi = sorted(smi.split("."), key=len)[-1]
        out.append(smi)
    return out


class Sampler:
    def __init__(self, path=None, forward_op=None, model=None, **kwargs):
        self.model = model if model is not None else load_model_from_path(path)
        self.slicer = Slicer()
        self.dot_index = self.model.tokenizer('.')['input_ids'][1]
        self.pad_index = self.model.tokenizer.pad_token_id
        self.mdlm = self.model.mdlm
        self.mdlm.to_device(self.model.device)

        # Trajectory tracking (populated by subclasses during generation)
        self._trajectory = []
        self._cumul_reward_calls = 0
        self._cumul_forward_passes = 0
        self._best_reward = float("-inf")

    @torch.no_grad()
    def _tweedie_x0(self, x):
        """One-shot Tweedie estimate of x0 from xt.
        Single forward pass; fill every masked position with argmax over vocab.
        Returns (pseudo_clean_tensor, fp_count)."""
        attention_mask = x != self.pad_index
        logits = self.model(x, attention_mask)
        pseudo_x0 = x.clone()
        mask_positions = (x == self.model.mask_index)
        if mask_positions.any():
            pseudo_x0[mask_positions] = logits[mask_positions].argmax(-1)
        return pseudo_x0, x.shape[0]

    @torch.no_grad()
    def generate(self, x, softmax_temp=1.2, randomness=2, fix=True, gamma=0, w=2, **kwargs):
        x = x.to(self.model.device)
        num_steps = max(self.mdlm.get_num_steps_confidence(x), 2)
        attention_mask = x != self.pad_index
        
        for i in range(num_steps):
            logits = self.model(x, attention_mask)

            if gamma and w:
                x_poor = x.clone()
                context_tokens = (x_poor[0] != self.model.bos_index).to(int) * \
                    (x_poor[0] != self.model.eos_index).to(int) * \
                    (x_poor[0] != self.model.mask_index).to(int) * \
                    (x_poor[0] != self.pad_index).to(int)
                context_token_ids = context_tokens.nonzero(as_tuple=True)[0].tolist()
                # mask 100 * gamma % of the context (given fragments) tokens
                num_mask_poor = int(context_tokens.sum() * gamma)
                mask_idx_poor = random.sample(context_token_ids, num_mask_poor)
                x_poor[:, mask_idx_poor] = self.model.mask_index
                logits_poor = self.model(x_poor, attention_mask=attention_mask)
                logits = w * logits + (1 - w) * logits_poor

            x = self.mdlm.step_confidence(logits, x, i, num_steps, softmax_temp, randomness)
            
        # decode to SAFE strings
        samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
        # convert to SMILES strings
        if self.model.config.training.get('use_bracket_safe'):
            samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
        else:
            samples = [safe_to_smiles(s, fix=fix) for s in samples]
        # remove None and take the largest
        samples = [sorted(s.split('.'), key=len)[-1] for s in samples if s]
        return samples

    def _insert_mask(self, x, num_samples, min_add_len=18, **kwargs):
        with open(os.path.join(ROOT_DIR, 'data/len.pk'), 'rb') as f:
            seq_len_list = pickle.load(f)
        
        x = x[0]
        x_new = []
        for _ in range(num_samples):
            add_seq_len = max(random.choice(seq_len_list) - len(x), min_add_len)
            x_new.append(torch.hstack([x[:-1],
                                      torch.full((add_seq_len,), self.model.mask_index),
                                      x[-1:]]))
        pad_len = max([len(xx) for xx in x_new])
        x_new = [torch.hstack([xx,torch.full((pad_len - len(xx),), self.pad_index)]) for xx in x_new]
        return torch.stack(x_new)

    # ── Trajectory tracking ───────────────────────────────────────

    def _reset_trajectory(self):
        """Call at the start of de_novo_generation to reset tracking state."""
        from time import time
        self._trajectory = []
        self._cumul_reward_calls = 0
        self._cumul_forward_passes = 0
        self._best_reward = float("-inf")
        self._round_id = 0
        self._t0 = time()
        self._scored_pool = []  # list of (smiles, score)
        self._unique_smiles = set()
        self._snapshot_hours = set()
        self._snapshots = {}
        self._last_logged_calls = 0  # for 10-call interval logging
        # Oracle-call milestones: set via sampler.snapshot_call_milestones
        # before calling de_novo_generation
        if not hasattr(self, 'snapshot_call_milestones'):
            self.snapshot_call_milestones = set()
        self._snapshot_calls_done = set()

    def _log_point(self, reward_calls_delta, fp_delta, best_reward_this_batch,
                   smiles_batch=None, scores_batch=None, **extras):
        """Record trajectory data. Emits a row every 10 cumulative oracle calls.

        extras: method-specific fields (candidates_kept, cum_simulations, etc.)
        """
        from time import time
        self._cumul_reward_calls += reward_calls_delta
        self._cumul_forward_passes += fp_delta
        self._best_reward = max(self._best_reward, best_reward_this_batch)
        wall = time() - self._t0 if hasattr(self, '_t0') else 0.0

        # Track scored molecules for running metrics + snapshots
        if smiles_batch is not None and scores_batch is not None:
            for smi, sc in zip(smiles_batch, scores_batch):
                if smi and sc is not None:
                    self._scored_pool.append((sc, smi))
                    self._unique_smiles.add(smi)

        # Emit trajectory row every 10 oracle calls
        if self._cumul_reward_calls - self._last_logged_calls >= 10:
            self._last_logged_calls = (self._cumul_reward_calls // 10) * 10
            self._round_id += 1

            # Compute running metrics from scored pool
            if self._scored_pool:
                all_scores = sorted([sc for sc, _ in self._scored_pool], reverse=True)
                n = len(all_scores)
                top10_mean = sum(all_scores[:max(1, n // 10)]) / max(1, n // 10)
                top1_mean = sum(all_scores[:max(1, n // 100)]) / max(1, n // 100)
            else:
                top10_mean = 0.0
                top1_mean = 0.0

            row = {
                "wall_sec": wall,
                "round_id": self._round_id,
                "cum_oracle_calls": self._cumul_reward_calls,
                "cum_forward": self._cumul_forward_passes,
                "running_best": self._best_reward,
                "top10_mean": top10_mean,
                "top1_mean": top1_mean,
                "cum_unique_molecules": len(self._unique_smiles),
            }
            row.update(extras)
            self._trajectory.append(row)
            print(f"  [{self.__class__.__name__}] {wall:.0f}s  oracle={self._cumul_reward_calls}  "
                  f"best={self._best_reward:.4f}  top10={top10_mean:.4f}  "
                  f"unique={len(self._unique_smiles)}", flush=True)

        # Hourly snapshots
        elapsed_hr = int(wall / 3600)
        if elapsed_hr > 0 and elapsed_hr not in self._snapshot_hours:
            self._snapshot_hours.add(elapsed_hr)
            self._snapshots[f"{elapsed_hr}hr"] = self._build_snapshot()

        # Oracle-call-based snapshots
        for milestone in self.snapshot_call_milestones:
            if (milestone not in self._snapshot_calls_done
                    and self._cumul_reward_calls >= milestone):
                self._snapshot_calls_done.add(milestone)
                self._snapshots[f"{milestone}calls"] = self._build_snapshot()

    def _build_snapshot(self):
        """Build a snapshot dict from the current scored pool."""
        from time import time
        pool_sorted = sorted(self._scored_pool, reverse=True)
        seen = set()
        unique = []
        for sc, smi in pool_sorted:
            if smi not in seen:
                seen.add(smi)
                unique.append({"smiles": smi, "score": sc})
        wall = time() - self._t0 if hasattr(self, '_t0') else 0.0
        all_scores = [d["score"] for d in unique]
        n = len(all_scores)
        top10_n = max(1, n // 10)
        snapshot = {
            "molecules": unique,
            "best_FA": all_scores[0] if all_scores else None,
            "top10_mean_FA": sum(all_scores[:10]) / min(10, n) if n else None,
            "top10pct_mean_FA": sum(all_scores[:top10_n]) / top10_n if n else None,
            "unique_molecule_count": n,
            "cum_oracle_calls": self._cumul_reward_calls,
            "wall_sec": wall,
            "tanimoto_diversity": _tanimoto_diversity([d["smiles"] for d in unique[:200]]),
        }
        return snapshot

    @property
    def trajectory(self):
        return list(self._trajectory)

    @property
    def snapshots(self):
        return dict(self._snapshots)

    @torch.no_grad()
    def de_novo_generation(self, num_samples=1, softmax_temp=0.8, randomness=0.5, min_add_len=40, timeout_sec=None, **kwargs):
        from time import time
        BATCH_SIZE = 256
        t0 = time()
        all_smiles = []
        remaining = num_samples
        while remaining > 0:
            if timeout_sec is not None and time() - t0 > timeout_sec:
                break
            bs = min(remaining, BATCH_SIZE)
            x = torch.hstack([torch.full((1, 1), self.model.bos_index),
                              torch.full((1, 1), self.model.eos_index)])
            x = self._insert_mask(x, bs, min_add_len=min_add_len)
            x = x.to(self.model.device)
            batch_smiles = self.generate(x, softmax_temp, randomness)
            all_smiles.extend(batch_smiles)
            remaining -= bs
            if len(all_smiles) % 1024 < BATCH_SIZE:
                print(f"  [baseline] {time()-t0:.0f}s  n={len(all_smiles)}", flush=True)
        return all_smiles
    
    def fragment_linking_onestep(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        if self.model.config.training.get('use_bracket_safe'):
            encoded_fragment = BracketSAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        else:
            encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        
        x = self.model.tokenizer([encoded_fragment + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma)
        samples = filter_by_substructure(samples, fragment)
        return samples
    
    def fragment_linking(self, fragment, num_samples=1, softmax_temp=1.2, randomness=2, gamma=0, min_add_len=30, **kwargs):
        encoded_fragment = sf.SAFEConverter(slicer=None).encoder(fragment, allow_empty=True)
        prefix, suffix = encoded_fragment.split('.')

        x = self.model.tokenizer([prefix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        prefix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma)

        x = self.model.tokenizer([suffix + '.'],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
        suffix_samples = self.generate(x, softmax_temp, randomness, gamma=gamma)
        
        samples = filter_by_substructure(mix_sequences(prefix_samples, suffix_samples,
                                                      *fragment.split('.'), num_samples), fragment)
        return samples
        
    def fragment_completion(self, fragment, num_samples=1, apply_filter=True, softmax_temp=1.2, randomness=2, gamma=0, **kwargs):
        if '*' not in fragment:     # superstructure generation
            cores = sf.utils.list_individual_attach_points(Chem.MolFromSmiles(fragment), depth=3)
            fragment = random.choice(cores)
            
        encoded_fragment = sf.SAFEConverter(ignore_stereo=True).encoder(fragment, allow_empty=True) + '.'
        x = self.model.tokenizer([encoded_fragment],
                                 return_tensors='pt',
                                 truncation=True,
                                 max_length=self.model.config.model.max_position_embeddings)['input_ids']
        x = self._insert_mask(x, num_samples)
        samples = self.generate(x, softmax_temp, randomness, gamma=gamma)

        if apply_filter:
            return filter_by_substructure(samples, fragment)
        return samples

    def mask_modification(self, smiles, min_len=30, **kwargs):
        encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
        x = self.model.tokenizer([encoded_smiles],
                                  return_tensors='pt',
                                  truncation=True,
                                  max_length=self.model.config.model.max_position_embeddings)['input_ids']
        if x.shape[-1] < min_len:
            return self.addmask(smiles, num_edit=min_len-x.shape[-1]+1, **kwargs)
        return self.remask(smiles, input_ids=x, **kwargs)

    def addmask(self, smiles, num_edit=3, **kwargs):
        try:
            samples = self.fragment_completion(smiles, mask_len=num_edit, apply_filter=False, **kwargs)
        except:
            return smiles
        if samples:
            return samples[0]
        return smiles
    
    def remask(self, smiles, input_ids=None, **kwargs):
        x = input_ids
        if x is None:
            encoded_smiles = sf.SAFEConverter(slicer=self.slicer, ignore_stereo=True).encoder(smiles, allow_empty=True)
            x = self.model.tokenizer([encoded_smiles],
                                     return_tensors='pt',
                                     truncation=True,
                                     max_length=self.model.config.model.max_position_embeddings)['input_ids']
        
        # fragment mask replacement
        special_token_idx = [0] + (x[0] == self.dot_index).nonzero(as_tuple=True)[0].tolist() + [len(x[0]) - 1]
        frag_idx = random.randint(0, len(special_token_idx) - 2)
        mask_start_idx = special_token_idx[frag_idx] + 1
        mask_end_idx = special_token_idx[frag_idx + 1]
        num_insert_mask = random.randint(5, 15)
        num_insert_mask = min(num_insert_mask,
                              self.model.config.model.max_position_embeddings - x.shape[-1] + mask_end_idx - mask_start_idx)
        x = torch.hstack([x[:, :mask_start_idx],
                          torch.full((1, num_insert_mask), self.model.mask_index),
                          x[:, mask_end_idx:]])
        samples = self.generate(x, **kwargs)
        if samples:
            return samples[0]
        return smiles
