import os
import warnings
import random
import math
import torch
import safe as sf
from rdkit import Chem
from rdkit.Chem import Descriptors

from genmol.utils.utils_chem import safe_to_smiles
from genmol.utils.bracket_safe_converter import bracketsafe2safe
from genmol.sampler import Sampler


os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")


class DAPSSampler(Sampler):
	"""
	Decoupled Annealing Posterior Sampling (DAPS) for MDLM-based GenMol.

	This implementation focuses on unconditional (de novo) generation.
	It alternates MDLM denoising steps with a controlled re-masking
	schedule to encourage exploration early and stabilization late.
	"""

	def __init__(
		self,
		path,
		num_steps=20,
		forward_op=None,
		alpha=1.0,
		mh_steps=100,
		max_mutations=1,
		remask_max=0.6,
		remask_min=0.05,
		remask_schedule="linear",
		seed=None,
	):
		super().__init__(path)
		self.num_steps = max(int(num_steps), 2)
		self.forward_op = forward_op or MolecularWeightForwardOp()
		self.alpha = float(alpha)
		self.mh_steps = max(int(mh_steps), 0)
		self.max_mutations = max(int(max_mutations), 1)
		self.remask_max = float(remask_max)
		self.remask_min = float(remask_min)
		self.remask_schedule = remask_schedule
		if seed is not None:
			random.seed(seed)
			torch.manual_seed(seed)

	def _mask_fraction(self, step):
		if self.num_steps <= 1:
			return 0.0
		progress = step / (self.num_steps - 1)

		if self.remask_schedule == "cosine":
			# Start high, end low
			return self.remask_min + 0.5 * (self.remask_max - self.remask_min) * (1 + math.cos(math.pi * progress))

		# Default: linear decay from max to min
		return self.remask_max - (self.remask_max - self.remask_min) * progress

	def _remask(self, x, mask_fraction):
		if mask_fraction <= 0:
			return x

		x = x.clone()
		for i in range(x.shape[0]):
			row = x[i]
			maskable = (
				(row != self.pad_index)
				& (row != self.model.bos_index)
				& (row != self.model.eos_index)
			)
			idx = maskable.nonzero(as_tuple=True)[0]
			if idx.numel() == 0:
				continue
			k = max(1, int(round(idx.numel() * mask_fraction)))
			perm = torch.randperm(idx.numel(), device=row.device)
			selected = idx[perm[:k]]
			row[selected] = self.model.mask_index
			x[i] = row
		return x

	def _decode(self, x, fix=True):
		samples = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
		if self.model.config.training.get("use_bracket_safe"):
			samples = [safe_to_smiles(bracketsafe2safe(s), fix=fix) for s in samples]
		else:
			samples = [safe_to_smiles(s, fix=fix) for s in samples]
		samples = [sorted(s.split("."), key=len)[-1] for s in samples if s]
		return samples

	def _decode_safe(self, x):
		return self.model.tokenizer.batch_decode(x, skip_special_tokens=True)

	def _encode_safe(self, safe_str, length):
		encoded = self.model.tokenizer(
			[safe_str],
			return_tensors="pt",
			truncation=True,
			max_length=length,
		)["input_ids"][0]
		if encoded.shape[0] < length:
			pad = torch.full((length - encoded.shape[0],), self.pad_index, device=encoded.device)
			encoded = torch.hstack([encoded, pad])
		return encoded

	def _get_safe_charset(self):
		if hasattr(self, "_safe_charset") and self._safe_charset:
			return self._safe_charset

		charset = set()
		vocab = None
		if hasattr(self.model.tokenizer, "get_vocab"):
			try:
				vocab = self.model.tokenizer.get_vocab()
			except Exception:
				vocab = None
		if isinstance(vocab, dict):
			for tok in vocab.keys():
				if isinstance(tok, str) and len(tok) == 1:
					charset.add(tok)

		if not charset:
			charset = set(list("CNOPSFclbrI[]=#()1234567890+-@"))
		self._safe_charset = sorted(charset)
		return self._safe_charset

	def _mutate_safe_fragment(self, safe_str):
		fragments = [f for f in safe_str.split(".") if f]
		if not fragments:
			return safe_str
		idx = random.randrange(len(fragments))
		frag = fragments[idx]
		charset = self._get_safe_charset()
		if not frag:
			frag = random.choice(charset)
		else:
			pos = random.randrange(len(frag))
			new_char = frag[pos]
			for _ in range(10):
				candidate = random.choice(charset)
				if candidate != frag[pos]:
					new_char = candidate
					break
			frag = frag[:pos] + new_char + frag[pos + 1 :]

		fragments[idx] = frag
		return ".".join(fragments)

	def _decode_keep_none(self, x, fix=True):
		decoded = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
		out = []
		for s in decoded:
			try:
				if self.model.config.training.get("use_bracket_safe"):
					smi = safe_to_smiles(bracketsafe2safe(s), fix=fix)
				else:
					smi = safe_to_smiles(s, fix=fix)
			except Exception:
				smi = None
			if smi:
				smi = sorted(smi.split("."), key=len)[-1]
			out.append(smi)
		return out

	def _propose_tokens(self, x):
		x = x.clone()
		safe_strings = self._decode_safe(x)
		seq_len = x.shape[1]

		for i, safe_str in enumerate(safe_strings):
			mutated = safe_str
			for _ in range(self.max_mutations):
				mutated = self._mutate_safe_fragment(mutated)
			encoded = self._encode_safe(mutated, seq_len).to(x.device)
			x[i] = encoded

		return x

	def _mh_step(self, x):
		if self.forward_op is None or self.mh_steps <= 0:
			return x

		for _ in range(self.mh_steps):
			proposal = self._propose_tokens(x)
			cur_smiles = self._decode_keep_none(x)
			prop_smiles = self._decode_keep_none(proposal)

			cur_scores = self.forward_op(cur_smiles)
			prop_scores = self.forward_op(prop_smiles)

			log_ratio = self.alpha * (prop_scores - cur_scores)
			log_ratio = torch.nan_to_num(log_ratio, nan=-1e9, neginf=-1e9, posinf=1e9)
			accept_prob = torch.clamp(torch.exp(log_ratio), max=1.0)
			u = torch.rand_like(accept_prob)
			accept = (u < accept_prob).unsqueeze(-1)
			x = torch.where(accept, proposal, x)

		return x

	@torch.no_grad()
	def de_novo_generation(self, num_samples=1, softmax_temp=0.8, randomness=0.5, min_add_len=40, **kwargs):
		# Prepare fully masked inputs
		x = torch.hstack(
			[
				torch.full((1, 1), self.model.bos_index),
				torch.full((1, 1), self.model.eos_index),
			]
		)
		x = self._insert_mask(x, num_samples, min_add_len=min_add_len)
		x = x.to(self.model.device)

		for i in range(self.num_steps):
			attention_mask = x != self.pad_index
			logits = self.model(x, attention_mask)
			x = self.mdlm.step_confidence(logits, x, i, self.num_steps, softmax_temp, randomness)
			x = self._mh_step(x)

			if i < self.num_steps - 1:
				mask_fraction = self._mask_fraction(i)
				x = self._remask(x, mask_fraction)

		return self._decode(x)


class MolecularWeightForwardOp:
	def __call__(self, smiles_list):
		scores = []
		for smi in smiles_list:
			if not smi:
				scores.append(float("-inf"))
				continue
			try:
				mol = Chem.MolFromSmiles(smi)
				if mol is None:
					scores.append(float("-inf"))
					continue
				scores.append(float(Descriptors.MolWt(mol)))
			except Exception:
				scores.append(float("-inf"))

		return torch.tensor(scores, dtype=torch.float32)
