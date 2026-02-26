import os
import sys

# Set environment variables before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["RDKIT_QUIET"] = "1"

import warnings
import random
import math
import torch
import numpy as np
import re
from tqdm import tqdm

# Suppress all warnings before importing libraries
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress RDKit/SAFE warnings by redirecting stderr during import
import io
import contextlib

@contextlib.contextmanager
def suppress_output():
    """Suppress both stdout and stderr at the OS level."""
    # Save the original file descriptors
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    
    # Save copies of the original stdout/stderr
    stdout_dup = os.dup(stdout_fd)
    stderr_dup = os.dup(stderr_fd)
    
    # Open devnull
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    
    # Redirect stdout and stderr to devnull
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.TextIOWrapper(os.fdopen(devnull_fd, 'wb'))
    sys.stderr = sys.stdout
    
    os.dup2(devnull_fd, stdout_fd)
    os.dup2(devnull_fd, stderr_fd)
    
    try:
        yield
    finally:
        # Flush before restoring
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Restore the original file descriptors
        os.dup2(stdout_dup, stdout_fd)
        os.dup2(stderr_dup, stderr_fd)
        
        # Close the duplicates
        os.close(stdout_dup)
        os.close(stderr_dup)
        
        # Restore Python's stdout/stderr
        sys.stdout = old_stdout
        sys.stderr = old_stderr

# Import SAFE, RDKit and datamol with suppressed output
with suppress_output():
	import safe as sf
	import datamol as dm
	from rdkit import Chem
	from rdkit.Chem import Descriptors

# Disable RDKit logging (suppresses PandasTools patch messages)
try:
	from rdkit import RDLogger
	RDLogger.DisableLog("rdApp.*")
except Exception:
	pass

from genmol.utils.utils_chem import safe_to_smiles
from genmol.utils.bracket_safe_converter import bracketsafe2safe
from genmol.sampler import Sampler

# Additional warning filters
warnings.filterwarnings("ignore", module="pandas")
warnings.filterwarnings("ignore", module="datamol")
warnings.filterwarnings("ignore", module="rdkit")
warnings.filterwarnings("ignore", module="transformers")
warnings.filterwarnings("ignore", module="safe")
warnings.filterwarnings("ignore", message=".*Failed to patch pandas.*")
warnings.filterwarnings("ignore", message=".*PandasTools.*")
warnings.filterwarnings("ignore", message=".*early_stopping.*")
warnings.filterwarnings("ignore", message=".*generation flags.*")


class SafeFragmentLinker:
	"""
	Handles SAFE string fragment linking/unlinking operations.
	Works directly in SAFE space with explicit attachment points.
	Ensures fragments are attachment-compatible when adding or swapping.
	Generates novel fragments using fragment_completion (inspired by genmol's generate/attach).
	"""

	def __init__(self, seed=42, sampler=None):
		"""
		Initialize the linker.

		Args:
			seed: Random seed for reproducibility
			sampler: Optional Sampler object for generating novel fragments
					If provided, add/swap will generate new fragments instead of selecting from pool
		"""
		with suppress_output():
			self.converter = sf.SAFEConverter(slicer=None)
			# Cache SAFEDesign model for motif extension to avoid reloading
			self.designer = sf.SAFEDesign.load_default(verbose=False)
		self.sampler = sampler
		self.rng = np.random.RandomState(seed)
		random.seed(seed)

	def _suppress_safe_output(self):
		"""Context manager to suppress SAFE package stdout/stderr."""
		return suppress_output()

	def encode_smiles_to_safe(self, smiles: str) -> str:
		"""Convert SMILES to SAFE representation."""
		try:
			with self._suppress_safe_output():
				safe_str = self.converter.encoder(smiles, allow_empty=True)
			return safe_str if safe_str else None
		except Exception:
			return None

	def decode_safe_to_smiles(self, safe_str: str) -> str:
		"""Convert SAFE back to SMILES."""
		try:
			with self._suppress_safe_output():
				smiles = self.converter.decoder(safe_str)
			return smiles if smiles else None
		except Exception:
			return None

	def split_safe_fragments(self, safe_str: str) -> list:
		"""Split SAFE string by dots to get individual fragments."""
		if not safe_str:
			return []
		fragments = [f.strip() for f in safe_str.split(".") if f.strip()]
		return fragments

	def join_safe_fragments(self, fragments: list) -> str:
		"""Join fragments with dots."""
		return ".".join(f for f in fragments if f)

	def get_attachment_points(self, safe_str: str) -> list:
		"""Extract all attachment point numbers from SAFE string."""
		numbers = re.findall(r"\d+", safe_str)
		return sorted(set(int(n) for n in numbers))

	def _build_fragment_graph(self, fragments: list) -> list:
		"""Build fragment adjacency based on shared attachment point numbers."""
		ap_to_frags = {}
		for idx, frag in enumerate(fragments):
			for ap in self.get_attachment_points(frag):
				ap_to_frags.setdefault(ap, []).append(idx)

		adj = [set() for _ in fragments]
		for frags in ap_to_frags.values():
			if len(frags) < 2:
				continue
			for i in range(len(frags)):
				for j in range(i + 1, len(frags)):
					a, b = frags[i], frags[j]
					adj[a].add(b)
					adj[b].add(a)

		return adj

	def _is_connected(self, fragments: list) -> bool:
		"""Check if fragments form a single connected component via attachment points."""
		if len(fragments) <= 1:
			return True
		adj = self._build_fragment_graph(fragments)

		# If there are no edges, it's disconnected
		if all(len(neigh) == 0 for neigh in adj):
			return False

		visited = set()
		stack = [0]
		while stack:
			node = stack.pop()
			if node in visited:
				continue
			visited.add(node)
			for nbr in adj[node]:
				if nbr not in visited:
					stack.append(nbr)

		return len(visited) == len(fragments)

	def generate_novel_fragment(self, template_fragment: str = None) -> str:
		#TODO: have a diverse pool of fragments to sample and no aromatic cores defaulting
		"""
		Generate a novel fragment using the sampler (inspired by genmol's generate/attach pattern).

		If sampler is available:
		  - Use fragment_completion to generate new molecules around a template
		  - Extract a fragment from the generated molecule
		If no sampler available:
		  - Generate a random small SAFE fragment

		Args:
			template_fragment: Optional template fragment to condition generation on

		Returns:
			A novel SAFE fragment string
		"""
		if self.sampler is not None:
			try:
				if template_fragment:
					samples = self.sampler.fragment_completion(
						template_fragment,
						num_samples=1,
						apply_filter=True,
					)
				else:
					#TODO: replace this with random selection among known templates?
					samples = self.sampler.de_novo_generation(num_samples=1)

				if samples and samples[0]:
					safe_str = self.encode_smiles_to_safe(samples[0])
					if safe_str:
						frags = self.split_safe_fragments(safe_str)
						if frags:
							return frags[0]
			except Exception:
				pass

		aromatic_cores = [
			"c1ccccc1",
			"c1ccncc1",
			"c1cccnc1",
			"c1ccsc1",
			"c1ccoc1",
			"c1ccc2ccccc2c1",
		]

		core = random.choice(aromatic_cores)
		frag_with_ap = core + str(random.randint(1, 3))
		return frag_with_ap

	def identify_extension_points(self, smiles: str) -> list:
		"""
		Identify extension points in a molecule where new fragments can be added.
		Extension points are positions with available valence for bonding.
		"""
		try:
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				return []

			extension_points = []
			pt = Chem.GetPeriodicTable()

			for atom in mol.GetAtoms():
				explicit_val = atom.GetExplicitValence()
				implicit_val = atom.GetImplicitValence()
				total_val = explicit_val + implicit_val

				allowed_valences = pt.GetValenceList(atom.GetAtomicNum())
				max_val = max(allowed_valences) if allowed_valences else 4

				available_bonds = max_val - total_val
				if available_bonds > 0:
					emol = Chem.EditableMol(mol)
					dummy_idx = emol.AddAtom(Chem.Atom(0))
					emol.AddBond(atom.GetIdx(), dummy_idx, Chem.BondType.SINGLE)
					mol_with_dummy = emol.GetMol()
					smiles_with_dummy = Chem.MolToSmiles(mol_with_dummy)

					extension_points.append({
						"atom_idx": atom.GetIdx(),
						"atom_symbol": atom.GetSymbol(),
						"available_valence": available_bonds,
						"with_dummy": smiles_with_dummy,
					})

			return extension_points
		except Exception:
			return []

	def add_with_motif_extension(self, motif_smiles: str, n_candidates: int = 5) -> str:
		"""
		Perform motif extension using SAFEDesign.motif_extension() API.
		"""
		try:
			with self._suppress_safe_output():
				mol = dm.to_mol(motif_smiles)
			if mol is None:
				return None

			with self._suppress_safe_output():
				from safe import utils
				motifs_with_aps = list(utils.list_individual_attach_points(mol))
			if not motifs_with_aps:
				return None

			motif_with_ap = random.choice(motifs_with_aps)
			try:
				current_length = len(motif_with_ap)
				candidates = self.designer.motif_extension(
						motif=motif_with_ap,
						n_samples_per_trial=n_candidates,
						n_trials=1,
						sanitize=True,
						do_not_fragment_further=True,
						random_seed=42,
						# min_length=current_length + 1,
						# max_length=current_length + 10,
					)
			except Exception:
				return None

			if not candidates:
				return None

			valid_candidates = []
			for candidate in candidates:
				if candidate and candidate != motif_smiles:
					mol_cand = dm.to_mol(candidate)
					if mol_cand is not None:
						if mol_cand.GetNumAtoms() > mol.GetNumAtoms():
							valid_candidates.append(candidate)

			if not valid_candidates:
				return None

			best_smiles = valid_candidates[0]
			return best_smiles
		except Exception:
			return None

	def remove_fragment(self, safe_multi: str) -> str:
		"""Remove a terminal fragment, ensuring the decoded SMILES stays contiguous."""
		fragments = self.split_safe_fragments(safe_multi)
		
		if len(fragments) <= 1:
			return safe_multi
		
		end_indices = [0, len(fragments) - 1]
		random.shuffle(end_indices)
		
		for idx_to_remove in end_indices:
			remaining = [frag for i, frag in enumerate(fragments) if i != idx_to_remove]
			new_safe = self.join_safe_fragments(remaining)
			smiles = self.decode_safe_to_smiles(new_safe)
			if smiles and '.' not in smiles:
				return new_safe
		
		return safe_multi


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
			forward_op=None,
			num_steps=50,
			alpha=100.0,
			mh_steps=2,
			max_mutations=1,
			remask_max=0.6,
			remask_min=0.05,
			remask_schedule="linear",
			ode_steps=20,
			seed=None,
			**kwargs,
	):
		super().__init__(path, forward_op=forward_op, **kwargs)
		self.forward_op = forward_op #or MolecularWeightForwardOp()
		self.num_steps = max(int(num_steps), 1)
		self.alpha = float(alpha)
		self.mh_steps = max(int(mh_steps), 0)
		self.max_mutations = max(int(max_mutations), 1)
		self.remask_max = float(remask_max)
		self.remask_min = float(remask_min)
		self.remask_schedule = remask_schedule
		self.ode_steps = max(int(ode_steps), 2)
		if seed is not None:
			random.seed(seed)
			torch.manual_seed(seed)
		
		# Create timestep schedule (descending from high noise to low noise)
		# Following DAPS paper: start from high t and anneal to t=0
		self.timesteps = torch.linspace(1.0, 0.0, self.num_steps + 1)[:-1]
		
		#TODO: create a standard sampler?
		#sampler = Sampler(path)
		self._safe_linker = SafeFragmentLinker(seed=seed or 42, sampler=None)

	def _reverse_diffusion(self, xt, t_start):
		"""
		Perform reverse diffusion from noisy xt to predicted clean x0.
		Uses iterative denoising steps similar to ODE sampling.
		
		Args:
			xt: Current noisy tokens
			t_start: Starting timestep (noise level)
			
		Returns:
			x0hat: Predicted clean tokens
		"""
		x = xt.clone()
		# Create sub-timesteps for ODE solver from t_start down to 0
		sub_timesteps = torch.linspace(float(t_start), 0.0, self.ode_steps)
		
		for i in range(len(sub_timesteps) - 1):
			t_cur = sub_timesteps[i]
			t_next = sub_timesteps[i + 1]
			
			# Get model prediction at current timestep
			attention_mask = x != self.pad_index
			logits = self.model(x, attention_mask)
			
			# Use confidence-based sampling to denoise
			# Use low temperature for deterministic denoising
			x = self.mdlm.step_confidence(
				logits, x, i, len(sub_timesteps) - 1,
				0.1,  # softmax_temp - low temperature for more deterministic
				0.1   # randomness
			)
		return x

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
		decoded = []
		for s in samples:
			if not s:
				continue
			# Try direct SAFE decoding first
			smi = self._safe_linker.decode_safe_to_smiles(s)
			# If that fails, try with bracketsafe2safe conversion
			if not smi:
				try:
					converted = bracketsafe2safe(s)
					smi = self._safe_linker.decode_safe_to_smiles(converted)
				except Exception:
					smi = None
			if smi:
				decoded.append(sorted(smi.split("."), key=len)[-1])
		return decoded

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
		"""
		Mutate a SAFE string using either motif extension or fragment removal.
		
		Strategy (from notebook):
		- 50% probability: Use motif extension (SAFEDesign model with n_candidates=5)
		- 50% probability: Remove a terminal fragment (only if multiple fragments exist)
		  Falls back to adding a random fragment if removal fails or not applicable.
		"""
		# Decode once at the beginning to save time
		motif_smiles = self._safe_linker.decode_safe_to_smiles(safe_str)
		if not motif_smiles:
			return safe_str  # Return original if decode fails
		
		if random.random() < 0.5: 
			# Motif extension approach (50%)
			extended_smiles = self._safe_linker.add_with_motif_extension(
				motif_smiles, 
				n_candidates=5
			)
			if extended_smiles:
				extended_safe = self._safe_linker.encode_smiles_to_safe(extended_smiles)
				if extended_safe:
					return extended_safe
		
		# Remove Fragment approach (50% or if motif extension failed)
		# Re-encode to get proper SAFE format
		safe_orig = self._safe_linker.encode_smiles_to_safe(motif_smiles)
		if safe_orig:
			safe_frags = self._safe_linker.split_safe_fragments(safe_orig)
			if len(safe_frags) > 1:
				safe_removed = self._safe_linker.remove_fragment(safe_orig)
				# Skip expensive validation, just return if removal succeeded
				if safe_removed and safe_removed != safe_orig:
					return safe_removed
		
		# Fallback: add random fragment if removal didn't work or not applicable
		safe_frags = self._safe_linker.split_safe_fragments(safe_str)
		new_frag = self._safe_linker.generate_novel_fragment()
		return self._safe_linker.join_safe_fragments(safe_frags + [new_frag])

	def _decode_keep_none(self, x, fix=True, debug=False):
		decoded = self.model.tokenizer.batch_decode(x, skip_special_tokens=True)
		out = []
		failed_conversions = []
		for idx, s in enumerate(decoded):
			smi = None
			error_msg = None
			# Try direct SAFE decoding first
			try:
				smi = self._safe_linker.decode_safe_to_smiles(s)
			except Exception as e:
				error_msg = f"primary: {str(e)}"
				smi = None
			# If that fails, try with bracketsafe2safe conversion
			if not smi:
				try:
					converted = bracketsafe2safe(s)
					smi = self._safe_linker.decode_safe_to_smiles(converted)
					if not error_msg:
						error_msg = "primary failed silently, fallback succeeded"
				except Exception as e:
					if error_msg:
						error_msg += f"; fallback: {str(e)}"
					else:
						error_msg = f"fallback: {str(e)}"
			if smi:
				smi = sorted(smi.split("."), key=len)[-1]
			else:
				failed_conversions.append((idx, s[:80], error_msg))
			out.append(smi)
		
		if failed_conversions and debug:
			print(f"\n[_decode_keep_none] {len(failed_conversions)}/{len(decoded)} SAFE→SMILES conversions failed:")
			for idx, safe_str, err in failed_conversions[:5]:  # Show first 5
				print(f"  [{idx}] SAFE: {safe_str}...")
				print(f"       Error: {err}")
			if len(failed_conversions) > 5:
				print(f"  ... and {len(failed_conversions) - 5} more")
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

	def _mh_step(self, x, pbar=None):
		if self.forward_op is None or self.mh_steps <= 0:
			if pbar is not None:
				pbar.update(max(1, self.mh_steps))
			return x, None

		accept_sum = 0.0
		valid_sum = 0.0
		reward_sum = 0.0
		log_ratio_sum = 0.0
		steps = 0
		
		# Track valid proposals
		total_proposals = 0
		valid_safe_proposals = 0
		valid_smiles_proposals = 0
		
		for _ in range(self.mh_steps):
			proposal = self._propose_tokens(x)
			cur_smiles = self._decode_keep_none(x, debug=False)
			prop_smiles = self._decode_keep_none(proposal, debug=True)
			
			# If proposal SMILES are invalid, keep the original molecule
			valid_mask = torch.tensor([s is not None for s in prop_smiles], device=x.device)
			if valid_mask.numel() == x.size(0):
				invalid_mask = ~valid_mask
				if invalid_mask.any():
					proposal = proposal.clone()
					proposal[invalid_mask] = x[invalid_mask]
					for idx, is_valid in enumerate(valid_mask.tolist()):
						if not is_valid:
							prop_smiles[idx] = cur_smiles[idx]
			
			# Count valid SAFE strings (proposals that could be decoded)
			prop_safe = self._decode_safe(proposal)
			step_total = len(prop_safe)
			step_valid_safe = sum(s is not None and s != "" for s in prop_safe)
			total_proposals += step_total
			valid_safe_proposals += step_valid_safe
			
			# Track SMILES validity for debugging only
			valid_smiles_proposals += sum(s is not None for s in prop_smiles)

			cur_scores = self.forward_op(cur_smiles).to(x.device)
			prop_scores = self.forward_op(prop_smiles).to(x.device)

			log_ratio = self.alpha * (prop_scores - cur_scores)
			log_ratio = torch.nan_to_num(log_ratio, nan=-1e9, neginf=-1e9, posinf=1e9)
			accept_prob = torch.clamp(torch.exp(log_ratio), max=1.0)
			u = torch.rand_like(accept_prob).to(x.device)
			accept = (u < accept_prob).unsqueeze(-1).to(x.device)
			x = torch.where(accept, proposal, x)

			accept_sum += accept.float().mean().item()
			valid_sum += step_valid_safe / max(step_total, 1)
			reward_sum += torch.nan_to_num(prop_scores, nan=-1e9, neginf=-1e9, posinf=1e9).mean().item()
			log_ratio_sum += log_ratio.mean().item()
			steps += 1
			
			if pbar is not None:
				pbar.update(1)
		
		# Print validity statistics
		if total_proposals > 0:
			print(f"\n[MH Proposals] Total: {total_proposals} | "
				  f"Valid SAFE: {valid_safe_proposals}/{total_proposals} ({100*valid_safe_proposals/total_proposals:.1f}%) | "
				  f"Valid SMILES: {valid_smiles_proposals}/{total_proposals} ({100*valid_smiles_proposals/total_proposals:.1f}%)")

		stats = {
			"accept": accept_sum / max(steps, 1),
			"valid": valid_sum / max(steps, 1),
			"reward": reward_sum / max(steps, 1),
			"log_ratio": log_ratio_sum / max(steps, 1),
		}
		return x, stats
	
	def _visualize_molecules_after_mh(self, x):
		"""Visualize current molecules after MH step using RDKit."""
		import os
		from rdkit.Chem import Draw
		
		# Create output directory if it doesn't exist
		output_dir = "mh_visualizations"
		os.makedirs(output_dir, exist_ok=True)
		
		# Get current step number from existing files
		existing_files = [f for f in os.listdir(output_dir) if f.startswith("mh_step_") and f.endswith(".png")]
		step_num = len(existing_files) + 1
		
		# Decode molecules
		smiles_list = self._decode_keep_none(x, debug=False)
		
		# Create molecule objects
		mols = []
		valid_smiles = []
		for i, smi in enumerate(smiles_list):
			if smi:
				mol = Chem.MolFromSmiles(smi)
				if mol:
					mols.append(mol)
					valid_smiles.append(smi)
		
		if not mols:
			print(f"[MH Visualization] No valid molecules to visualize at step {step_num}")
			return
		
		# Create visualization
		try:
			legends = [f"Mol {i+1}" for i in range(len(mols))]
			img = Draw.MolsToGridImage(
				mols, 
				molsPerRow=min(4, len(mols)),
				subImgSize=(300, 300),
				legends=legends
			)
			
			# Save image
			output_path = os.path.join(output_dir, f"mh_step_{step_num:03d}.png")
			img.save(output_path)
			print(f"[MH Visualization] Saved {len(mols)} molecules to {output_path}")
		except Exception as e:
			print(f"[MH Visualization] Error creating image: {e}")


	@torch.no_grad()
	def de_novo_generation(self, num_samples=1, softmax_temp=0.8, randomness=0.5, min_add_len=40, **kwargs):
		"""
		De novo generation using Decoupled Annealing Posterior Sampling (DAPS).
		
		Algorithm:
		1. Start with fully noisy (masked) input xt
		2. For each annealing step i:
			a. Reverse diffusion: Denoise xt to get predicted clean state x0hat
			b. Metropolis-Hastings: Refine x0hat using forward operator (only if reward is defined)
			c. Forward diffusion: Add noise back to get xt for next step (except last step)
		3. Return final denoised samples
		"""
		# Prepare fully masked inputs (maximum noise)
		x = torch.hstack(
			[
				torch.full((1, 1), self.model.bos_index),
				torch.full((1, 1), self.model.eos_index),
			]
		)
		xt = self._insert_mask(x, num_samples, min_add_len=min_add_len)
		xt = xt.to(self.model.device)

		# Adjust progress bar based on whether we have a forward operator
		if self.forward_op is None:
			# No reward function: just do reverse diffusion
			total_iterations = self.num_steps
		else:
			# With reward function: reverse diffusion + MH steps
			total_iterations = self.num_steps * max(1, self.mh_steps)
		
		pbar = tqdm(total=total_iterations, desc="DAPS Sampling")

		for i in range(self.num_steps):
			t_current = self.timesteps[i]
			
			# Step 1: Reverse diffusion - denoise xt to get x0hat
			x0hat = self._reverse_diffusion(xt, t_current)
			
			# # Visualize current molecules
			self._visualize_molecules_after_mh(x0hat)
			
			# Step 2: Metropolis-Hastings - refine x0hat using forward operator (only if reward is defined)
			if self.forward_op is not None and self.alpha > 0 and self.mh_steps > 0:
				x0y, mh_stats = self._mh_step(x0hat, pbar)
			else:
				x0y, mh_stats = x0hat, None
				if self.forward_op is None:
					# No reward: just update progress bar for the step
					pbar.update(1)
				else:
					# Reward exists but alpha=0 or mh_steps=0: update for skipped MH steps
					pbar.update(max(1, self.mh_steps))
			
			# Log progress (use SAFE for display; SMILES only for scoring)
			safe_strings = self._decode_safe(x0y)
			if self.forward_op is not None:
				smiles = self._decode(x0y)
				scores = self.forward_op(smiles)
				mean_score = scores.mean().item()
				postfix = {"Mean Score": f"{mean_score:.2f}"} #"Step": i + 1, #"t": f"{t_current:.3f}"
				if mh_stats is not None:
					postfix.update({
						"MH acc": f"{mh_stats['accept']:.2f}",
						"MH valid": f"{mh_stats['valid']:.2f}",
						"MH logR": f"{mh_stats['log_ratio']:.2f}",
						"MH reward": f"{mh_stats['reward']:.2f}",
					})
				pbar.set_postfix(postfix)
			else:
				pbar.set_postfix({"Step": i + 1, "t": f"{t_current:.3f}"})
			
			# Step 3: Forward diffusion - add noise for next iteration
			if i < self.num_steps - 1:
				# Get next timestep
				t_next = self.timesteps[i + 1]
				# Add noise using forward process
				t_next_tensor = torch.full((num_samples,), float(t_next), device=xt.device)
				xt = self.mdlm.forward_process(x0y, t_next_tensor)
			else:
				# Last step: keep the refined clean state
				xt = x0y

		pbar.close()
		return self._decode_safe(xt)


class MolecularWeightForwardOp:
	def _smiles_error(self, smi):
		"""Return a reason string for invalid SMILES, or None if valid."""
		if not smi:
			return "empty"
		try:
			mol = Chem.MolFromSmiles(smi)
			if mol is not None:
				return None
		except Exception as exc:
			return f"MolFromSmiles error: {exc}"
		try:
			mol = Chem.MolFromSmiles(smi, sanitize=False)
			if mol is None:
				return "MolFromSmiles returned None"
			try:
				Chem.SanitizeMol(mol)
				return None
			except Exception as exc:
				return f"SanitizeMol error: {exc}"
		except Exception as exc:
			return f"MolFromSmiles(sanitize=False) error: {exc}"
		return "unknown"

	def __call__(self, smiles_list):
		scores = []
		invalid = []
		for smi in smiles_list:
			if not smi:
				scores.append(float("-inf"))
				invalid.append((smi, "empty"))
				continue
			try:
				mol = Chem.MolFromSmiles(smi)
				if mol is None:
					try:
						candidate = safe_to_smiles(bracketsafe2safe(smi), fix=True)
						mol = Chem.MolFromSmiles(candidate) if candidate else None
					except Exception:
						mol = None
				if mol is None:
					scores.append(float("-inf"))
					reason = self._smiles_error(smi)
					invalid.append((smi, reason))
					continue
				scores.append(float(Descriptors.MolWt(mol)))
			except Exception:
				scores.append(float("-inf"))
				reason = self._smiles_error(smi)
				invalid.append((smi, reason))
		#divide scores by 1000 to keep them in a reasonable range for exp/log operations in MH acceptance
		# if invalid:
		# 	print("\n[MW ForwardOp] Invalid SMILES detected:")
		# 	for smi, reason in invalid:
		# 		print(f"  - {smi}: {reason}")
		scores = [s / 1000.0 for s in scores]

		return torch.tensor(scores, dtype=torch.float32)
