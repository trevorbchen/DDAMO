#!/bin/bash
set -euo pipefail

# ===== Generate 1000 molecules unconditionally with MDLM, then score with FlashAffinity =====
#
# Usage:
#   cd /home/ariel/genmol
#   bash scripts/exps/denovo/run_uncond_flash_affinity.sh [RUN_NAME]
#
# This script:
#   1) Generates molecules unconditionally with pretrained MDLM
#   2) Runs the full native FlashAffinity pipeline (FABind+ docking → FlashAffinity scoring)

RUN_NAME="${1:-uncond_1000}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
PROT_ID="${PROT_ID:-2VT4}"
MODEL_PATH="${MODEL_PATH:-model_v2.ckpt}"

GENMOL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FA_ROOT="$GENMOL_ROOT/FlashAffinity"

echo "====== Config ======"
echo "  RUN_NAME:    $RUN_NAME"
echo "  NUM_SAMPLES: $NUM_SAMPLES"
echo "  PROT_ID:     $PROT_ID"
echo "  MODEL_PATH:  $MODEL_PATH"

# ===== Step 1: Generate molecules =====
echo ""
echo "====== Step 1: Generating $NUM_SAMPLES molecules unconditionally ======"

cd "$GENMOL_ROOT/scripts/exps/denovo"

python3 run_generation.py \
    sampler=uncond reward=none \
    num_samples="$NUM_SAMPLES" \
    model_path="$GENMOL_ROOT/$MODEL_PATH" \
    output_dir="outputs" \
    name=uncond

SAMPLES_CSV="$GENMOL_ROOT/scripts/exps/denovo/outputs/none/uncond/samples.csv"

if [[ ! -f "$SAMPLES_CSV" ]]; then
    echo "ERROR: Generation failed, no samples.csv found" >&2
    exit 1
fi

VALID_COUNT=$(tail -n+2 "$SAMPLES_CSV" | awk -F',' '{if ($1 != "") print}' | wc -l)
echo "  Generated samples: $SAMPLES_CSV ($VALID_COUNT valid)"

# ===== Step 2: Protein preprocessing (if not already done) =====
echo ""
echo "====== Step 2: Protein preprocessing ======"

cd "$FA_ROOT"
export PYTHONPATH="$FA_ROOT/src:${PYTHONPATH:-}"

DATA_DIR="$FA_ROOT/data/$RUN_NAME"
FABIND_WORK_DIR="$FA_ROOT/FABind_plus/protein_data_${RUN_NAME}"
LIGAND_DATA_DIR="$DATA_DIR"

mkdir -p "$DATA_DIR/pdb" "$DATA_DIR/repr" "$FABIND_WORK_DIR/pdb" "$FABIND_WORK_DIR/repr_files"

# Copy existing protein data
cp "$FA_ROOT/data/protein_test/pdb/${PROT_ID}.pdb" "$DATA_DIR/pdb/" 2>/dev/null || true
cp "$FA_ROOT/data/protein_test/pdb/${PROT_ID}.pdb" "$FABIND_WORK_DIR/pdb/" 2>/dev/null || true

# Copy ESM3 repr
if [[ -d "$FA_ROOT/data/protein_test/repr/esm3.lmdb" ]]; then
    cp -a "$FA_ROOT/data/protein_test/repr/esm3.lmdb" "$DATA_DIR/repr/" 2>/dev/null || true
fi

# Create prots.json from new_targets.json or extract from PDB
python3 - <<PY
import json, os
prot_id = "$PROT_ID"
data_dir = "$DATA_DIR"

# Try new_targets.json first
targets_file = "$FA_ROOT/data/protein_test/new_targets.json"
if os.path.exists(targets_file):
    with open(targets_file) as f:
        targets = json.load(f)
    if prot_id in targets:
        with open(os.path.join(data_dir, "prots.json"), "w") as f:
            json.dump({prot_id: targets[prot_id]}, f, indent=2)
        print(f"Wrote prots.json for {prot_id}")
        exit(0)

# Extract sequence from PDB
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
parser = PDBParser(QUIET=True)
structure = parser.get_structure(prot_id, os.path.join(data_dir, "pdb", f"{prot_id}.pdb"))
seq = ""
for model in structure:
    for chain in model:
        for residue in chain:
            if residue.id[0] == " ":
                seq += seq1(residue.get_resname())
        break
    break
with open(os.path.join(data_dir, "prots.json"), "w") as f:
    json.dump({prot_id: seq}, f, indent=2)
print(f"Extracted sequence ({len(seq)} AA) and wrote prots.json for {prot_id}")
PY

# Run FABind+ protein preprocessing (ESM2 features for docking)
if [[ ! -f "$FABIND_WORK_DIR/repr_files/processed_protein.pt" ]]; then
    echo "  Running FABind+ protein preprocessing..."
    cd "$FA_ROOT/FABind_plus/fabind"
    python3 inference_preprocess_protein_optimized.py \
        --pdb_file_dir "$FABIND_WORK_DIR/pdb" \
        --save_pt_dir "$FABIND_WORK_DIR/repr_files"
    cd "$FA_ROOT"
else
    echo "  FABind+ protein preprocessing already done, skipping."
fi

# ===== Step 3: Prepare ligand data =====
echo ""
echo "====== Step 3: Preparing ligand data ======"

export SMILES_CSV="$SAMPLES_CSV"
export PROT_ID DATA_DIR FABIND_WORK_DIR LIGAND_DATA_DIR

python3 - <<'PY'
import csv, json, os

smiles_csv = os.environ["SMILES_CSV"]
prot_id = os.environ["PROT_ID"]
ligand_data_dir = os.environ["LIGAND_DATA_DIR"]

os.makedirs(ligand_data_dir, exist_ok=True)

smiles_map = {}
id_list = []

with open(smiles_csv, "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    rows = list(reader)

header = [h.strip().lower() for h in rows[0]]
smiles_col = header.index("smiles")
data_rows = rows[1:]

for idx, row in enumerate(data_rows):
    if not row:
        continue
    smiles = row[smiles_col].strip()
    if not smiles:
        continue
    ligand_id = f"L{idx:06d}"
    smiles_map[ligand_id] = smiles
    id_list.append(f"{prot_id}_{ligand_id}")

for name, data in [("smiles.json", smiles_map), ("id.json", id_list)]:
    with open(os.path.join(ligand_data_dir, name), "w") as f:
        json.dump(data, f, indent=2)

with open(os.path.join(ligand_data_dir, "smiles.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["smiles", "ligand_id"])
    for lid, smi in smiles_map.items():
        w.writerow([smi, lid])

with open(os.path.join(ligand_data_dir, "data.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Smiles", "prot_id", "ligand_id"])
    for lid, smi in smiles_map.items():
        w.writerow([smi, prot_id, lid])

print(f"  Prepared {len(smiles_map)} ligands")
PY

# ===== Step 4: FABind+ molecule preprocessing =====
echo ""
echo "====== Step 4: FABind+ molecule preprocessing ======"

cd "$FA_ROOT/FABind_plus/fabind"
python3 inference_preprocess_mol_confs.py \
    --index_csv "$LIGAND_DATA_DIR/smiles.csv" \
    --save_mols_dir "$FABIND_WORK_DIR/repr_files" \
    --num_threads 0 \
    --resume

# ===== Step 5: FABind+ docking =====
echo ""
echo "====== Step 5: FABind+ docking inference ======"

python3 inference_regression_fabind.py \
    --ckpt "$FA_ROOT/FABind_plus/ckpt/fabind_plus_best_ckpt.bin" \
    --batch_size 4 \
    --post-optim \
    --write-mol-to-file \
    --sdf-output-path-post-optim "$LIGAND_DATA_DIR" \
    --index-csv "$LIGAND_DATA_DIR/data.csv" \
    --preprocess-dir "$FABIND_WORK_DIR/repr_files"

cd "$FA_ROOT"

# ===== Step 6: TorchDrug representation extraction =====
echo ""
echo "====== Step 6: TorchDrug representation extraction ======"

mkdir -p "$LIGAND_DATA_DIR/repr"

python3 src/affinity/data/repr/torchdrug.py \
    --input_json "$LIGAND_DATA_DIR/smiles.json" \
    --output_lmdb "$LIGAND_DATA_DIR/repr/torchdrug.lmdb" \
    --n_jobs -1

# ===== Step 7: FlashAffinity inference =====
echo ""
echo "====== Step 7: FlashAffinity inference ======"

torchrun --nproc_per_node=1 --rdzv_endpoint="localhost:29501" ./scripts/predict.py \
    --data "$LIGAND_DATA_DIR/id.json" \
    --task value \
    --structure "$DATA_DIR/pdb" \
    --structure_type pdb \
    --ligand "$LIGAND_DATA_DIR/ligand_sdf.lmdb" \
    --ligand_type sdf \
    --protein_repr "$DATA_DIR/repr/esm3.lmdb" \
    --ligand_repr "$LIGAND_DATA_DIR/repr/torchdrug.lmdb" \
    --distance_threshold 20.0 \
    --out_dir "$LIGAND_DATA_DIR" \
    --devices 1 \
    --affinity_checkpoint ./checkpoints/value_1.ckpt ./checkpoints/value_2.ckpt

echo ""
echo "====== Done! ======"
echo "  Generated molecules: $SAMPLES_CSV"
echo "  FlashAffinity results: $LIGAND_DATA_DIR/affinity_predictions_ensemble.json"
