#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test script: Visualize a single MH proposal step with before/after mutations.

This script:
1. Loads a DAPS sampler with a forward operator
2. Takes test molecules (from CSV)
3. Encodes them to tokens
4. Applies mutations via _mutate_safe_fragment (the MH proposal step)
5. Decodes back to SMILES
6. Visualizes before/after with detailed stats
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import safe as sf
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors

# Add genmol to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")))

from genmol.DAPS_sampler import DAPSSampler, MolecularWeightForwardOp
from genmol.utils.bracket_safe_converter import BracketSAFEConverter

warnings.filterwarnings("ignore")


CSV_DATA = """smiles,mol_wt
Cc1cncc(C(=O)N[C@@]2(C)CC[NH+](Cc3cncc(F)c3)C2)c1,329.399
Cc1ccccc1C[NH+]1CC[C@@H](NC(=O)C2CCOCC2)C1,303.4260000000001
CC1(C)CN(C(=O)c2csc(-c3ccccc3)n2)CCCO1,316.42600000000016
C[C@H](NC(=O)N1CCCCCC[NH2+]CC1)c1ccccc1F,308.42099999999994
CCCc1noc(CNC(=O)C[C@H]2CCOC2)n1,253.30199999999994
CC(C)(O)CNC(=O)c1ccc(NC(=O)c2ccc(C[NH+]3CCCC3)cc2)cc1,396.51100000000025
Cc1ccc(C(=O)NC2CC[NH+](CCC3CCOCC3)CC2)cc1,331.48000000000013
Cc1cccc(C(=O)N2CC[NH+](Cc3cccs3)CC2)c1,301.43500000000006
C[C@H]1C[C@@H](Cc2ccccc2)[NH+](OCc2ccccc2)C1,282.40700000000004
Cc1cccc(CNC(=O)C(=O)NCCCc2ccccc2)c1,310.397"""

DEFAULT_SMILES = [
    line.split(",")[0]
    for line in CSV_DATA.strip().splitlines()[1:]
    if line.strip()
]


def _get_safe_converter(use_bracket_safe: bool):
    if use_bracket_safe:
        return BracketSAFEConverter(slicer=None)
    return sf.SAFEConverter(slicer=None)


def _encode_smiles_batch(smiles_list, sampler, seq_len):
    """Encode SMILES to tokens."""
    use_bracket_safe = bool(sampler.model.config.training.get("use_bracket_safe"))
    converter = _get_safe_converter(use_bracket_safe)
    safe_strings = []
    
    for smi in smiles_list:
        try:
            safe_str = converter.encoder(smi, allow_empty=True)
        except Exception:
            safe_str = None
        if safe_str:
            safe_strings.append(safe_str)
    
    if not safe_strings:
        raise ValueError("No valid SAFE strings could be encoded from input SMILES.")
    
    encoded = [sampler._encode_safe(s, seq_len) for s in safe_strings]
    return torch.stack(encoded), safe_strings


def _decode_tokens_batch(tokens, sampler):
    """Decode tokens back to SMILES."""
    return sampler._decode_keep_none(tokens, fix=True)


def _get_molecular_properties(smiles):
    """Calculate molecular properties from SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        return {
            'atoms': mol.GetNumAtoms(),
            'bonds': mol.GetNumBonds(),
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
        }
    except Exception:
        return None


def visualize_proposal(before_smiles, after_smiles, proposals_safe, out_path, mols_per_row=3):
    """
    Create detailed visualization of MH proposal mutations.
    
    Shows:
    - Original molecules
    - Mutated SAFE strings
    - After-mutation SMILES
    - Property changes
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    n_mols = len(before_smiles)
    
    # Create figure with GridSpec for flexible layout
    fig = plt.figure(figsize=(15, 5*n_mols))
    gs = GridSpec(n_mols, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    for mol_idx in range(n_mols):
        before_smi = before_smiles[mol_idx]
        after_smi = after_smiles[mol_idx]
        safe_str = proposals_safe[mol_idx]
        
        # Get molecules
        before_mol = Chem.MolFromSmiles(before_smi) if before_smi else None
        after_mol = Chem.MolFromSmiles(after_smi) if after_smi else None
        
        # Get properties
        before_props = _get_molecular_properties(before_smi)
        after_props = _get_molecular_properties(after_smi)
        
        # ===== BEFORE =====
        ax_before = fig.add_subplot(gs[mol_idx, 0])
        if before_mol:
            AllChem.Compute2DCoords(before_mol)
            img = Draw.MolToImage(before_mol, size=(300, 300))
            ax_before.imshow(img)
        
        title_before = f"BEFORE (Mol {mol_idx+1})\n"
        if before_props:
            title_before += f"Atoms: {before_props['atoms']}, MW: {before_props['mw']:.1f}\n"
            title_before += f"LogP: {before_props['logp']:.2f}, HBD: {before_props['hbd']}"
        ax_before.set_title(title_before, fontsize=10, fontweight='bold', color='navy')
        ax_before.axis('off')
        
        # ===== SAFE STRING & MUTATION TYPE =====
        ax_info = fig.add_subplot(gs[mol_idx, 1])
        ax_info.axis('off')
        
        # Determine mutation type
        if before_smi == after_smi:
            mutation_type = "❌ NO MUTATION"
            color = 'red'
        elif after_smi is None:
            mutation_type = "❌ INVALID PROPOSAL"
            color = 'red'
        else:
            before_mol_obj = Chem.MolFromSmiles(before_smi)
            after_mol_obj = Chem.MolFromSmiles(after_smi)
            if before_mol_obj and after_mol_obj:
                before_atoms = before_mol_obj.GetNumAtoms()
                after_atoms = after_mol_obj.GetNumAtoms()
                if after_atoms > before_atoms:
                    mutation_type = f"✓ EXTENSION (+{after_atoms - before_atoms} atoms)"
                    color = 'green'
                elif after_atoms < before_atoms:
                    mutation_type = f"✓ REMOVAL (-{before_atoms - after_atoms} atoms)"
                    color = 'orange'
                else:
                    mutation_type = "✓ SUBSTITUTION"
                    color = 'blue'
            else:
                mutation_type = "? UNKNOWN"
                color = 'gray'
        
        info_text = f"MUTATION INFO\n"
        info_text += f"{'='*30}\n"
        info_text += f"{mutation_type}\n\n"
        info_text += f"SAFE String:\n"
        safe_display = safe_str if len(safe_str) < 60 else safe_str[:57] + "..."
        info_text += f"{safe_display}\n\n"
        
        # Fragment info
        fragments = safe_str.split('.')
        info_text += f"Fragments: {len(fragments)}\n"
        
        ax_info.text(0.05, 0.5, info_text, fontsize=9, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor=color, alpha=0.2))
        
        # ===== AFTER =====
        ax_after = fig.add_subplot(gs[mol_idx, 2])
        if after_mol:
            AllChem.Compute2DCoords(after_mol)
            img = Draw.MolToImage(after_mol, size=(300, 300))
            ax_after.imshow(img)
        else:
            ax_after.text(0.5, 0.5, 'INVALID', ha='center', va='center',
                         fontsize=14, color='red', fontweight='bold')
        
        title_after = f"AFTER (Mol {mol_idx+1})\n"
        if after_props:
            title_after += f"Atoms: {after_props['atoms']}, MW: {after_props['mw']:.1f}\n"
            title_after += f"LogP: {after_props['logp']:.2f}, HBD: {after_props['hba']}"
        ax_after.set_title(title_after, fontsize=10, fontweight='bold', color='darkgreen')
        ax_after.axis('off')
    
    fig.suptitle('MH Proposal Step: Before/After Visualization\n(50% Motif Extension + 50% Fragment Removal)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {out_path}")
    return fig


def main():
    parser = argparse.ArgumentParser(description="Test single MH proposal step and visualize.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="outputs/mh_proposal_test", help="Output folder")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of molecules to test")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length for SAFE tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*80)
    print("MH PROPOSAL STEP TEST: Single Mutation Visualization")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Num samples: {args.num_samples}")
    print()
    
    # Load sampler
    print("Loading sampler...")
    forward_op = MolecularWeightForwardOp()
    sampler = DAPSSampler(
        path=args.model_path,
        forward_op=forward_op,
        mh_steps=1,
        alpha=1.0,
        seed=args.seed,
    )
    
    # Select molecules
    test_smiles = DEFAULT_SMILES[:args.num_samples]
    print(f"✓ Sampler loaded")
    print(f"✓ Using {len(test_smiles)} test molecules")
    print()
    
    # Determine sequence length
    seq_len = args.seq_len
    if seq_len is None:
        # Calculate from test molecules
        use_bracket_safe = bool(sampler.model.config.training.get("use_bracket_safe"))
        converter = _get_safe_converter(use_bracket_safe)
        max_safe_len = 0
        for smi in test_smiles:
            try:
                safe_str = converter.encoder(smi, allow_empty=True)
                if safe_str:
                    max_safe_len = max(max_safe_len, len(safe_str))
            except Exception:
                pass
        seq_len = min(max_safe_len + 8, 256)
    
    # Encode
    print(f"Encoding molecules (seq_len={seq_len})...")
    x, safe_strings_original = _encode_smiles_batch(test_smiles, sampler, seq_len)
    print(f"✓ Encoded {x.shape[0]} molecules to tokens {x.shape}")
    print()
    
    # Show original molecules
    print("ORIGINAL MOLECULES:")
    print("-" * 80)
    for idx, (smi, safe_str) in enumerate(zip(test_smiles, safe_strings_original)):
        mol = Chem.MolFromSmiles(smi)
        atoms = mol.GetNumAtoms() if mol else "?"
        print(f"{idx+1}. SMILES: {smi}")
        print(f"   SAFE:   {safe_str}")
        print(f"   Atoms:  {atoms}")
    print()
    
    # Apply single proposal step (via _propose_tokens which uses _mutate_safe_fragment)
    print("Applying MH proposal mutations (50% motif extension + 50% removal)...")
    x_proposal = sampler._propose_tokens(x)
    print(f"✓ Generated proposals")
    print()
    
    # Decode
    print("Decoding proposals back to SMILES...")
    after_smiles = _decode_tokens_batch(x_proposal, sampler)
    print(f"✓ Decoded {len(after_smiles)} molecules")
    print()
    
    # Decode original to SAFE (for display)
    use_bracket_safe = bool(sampler.model.config.training.get("use_bracket_safe"))
    converter = _get_safe_converter(use_bracket_safe)
    proposals_safe = []
    for i in range(x_proposal.shape[0]):
        safe_str = sampler._decode_safe(x_proposal[i:i+1])
        proposals_safe.append(safe_str[0] if safe_str else "INVALID")
    
    # Show results
    print("PROPOSAL RESULTS:")
    print("-" * 80)
    for idx in range(len(test_smiles)):
        before = test_smiles[idx]
        after = after_smiles[idx]
        safe_prop = proposals_safe[idx]
        
        before_mol = Chem.MolFromSmiles(before)
        after_mol = Chem.MolFromSmiles(after) if after else None
        
        before_atoms = before_mol.GetNumAtoms() if before_mol else "?"
        after_atoms = after_mol.GetNumAtoms() if after_mol else "?"
        
        valid = "✓" if after else "✗"
        
        print(f"\nMolecule {idx+1}:")
        print(f"  Before: {before} ({before_atoms} atoms)")
        print(f"  After:  {after if after else 'INVALID'} ({after_atoms} atoms) {valid}")
        print(f"  SAFE:   {safe_prop[:70]}{'...' if len(safe_prop) > 70 else ''}")
        
        if after and before != after:
            before_obj = Chem.MolFromSmiles(before)
            after_obj = Chem.MolFromSmiles(after)
            if before_obj and after_obj:
                atom_change = after_obj.GetNumAtoms() - before_obj.GetNumAtoms()
                if atom_change > 0:
                    print(f"  Type:   EXTENSION (+{atom_change} atoms)")
                elif atom_change < 0:
                    print(f"  Type:   REMOVAL ({atom_change} atoms)")
                else:
                    print(f"  Type:   SUBSTITUTION")
    print()
    
    # Visualize
    print("Creating visualization...")
    viz_path = os.path.join(args.output_dir, "mh_proposal_before_after.png")
    visualize_proposal(test_smiles, after_smiles, proposals_safe, viz_path)
    
    # Summary statistics
    print()
    print("="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    valid_count = sum(1 for s in after_smiles if s is not None)
    changed_count = sum(1 for before, after in zip(test_smiles, after_smiles) 
                        if after and before != after)
    
    print(f"Total molecules: {len(test_smiles)}")
    print(f"Valid proposals: {valid_count}/{len(test_smiles)} ({100*valid_count/len(test_smiles):.1f}%)")
    print(f"Changed molecules: {changed_count}/{len(test_smiles)} ({100*changed_count/len(test_smiles):.1f}%)")
    
    # Mutation type breakdown
    extension_count = 0
    removal_count = 0
    for before, after in zip(test_smiles, after_smiles):
        if after and before != after:
            before_mol = Chem.MolFromSmiles(before)
            after_mol = Chem.MolFromSmiles(after)
            if before_mol and after_mol:
                if after_mol.GetNumAtoms() > before_mol.GetNumAtoms():
                    extension_count += 1
                elif after_mol.GetNumAtoms() < before_mol.GetNumAtoms():
                    removal_count += 1
    
    print(f"\nMutation types:")
    print(f"  Extensions (motif): {extension_count}")
    print(f"  Removals:          {removal_count}")
    print(f"  Substitutions:     {changed_count - extension_count - removal_count}")
    
    print()
    print(f"✓ All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
