#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Add src to Python path for module imports
export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"

# Set NCBI Entrez email (required for API access)
export ENTREZ_EMAIL="${ENTREZ_EMAIL:-user@example.com}"
export ENTREZ_API_KEY="${ENTREZ_API_KEY:-}"

# ===== User inputs (override via env vars) =====
SMILES_CSV="${SMILES_CSV:-$ROOT_DIR/data/input_smiles.csv}"
PROT_ID="${PROT_ID:-2VT4}"
RUN_ID="${RUN_ID:-${PROT_ID}_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/test}"  # Shared protein data directory
FABIND_WORK_DIR="${FABIND_WORK_DIR:-$ROOT_DIR/FABind_plus/protein_data}"  # Shared FABind protein directory
LIGAND_DATA_DIR="${LIGAND_DATA_DIR:-$DATA_DIR}"  # Shared data directory
OUT_DIR="${OUT_DIR:-$ROOT_DIR/value/test}"  # Output directory for predictions
FABIND_CKPT="${FABIND_CKPT:-$ROOT_DIR/FABind_plus/ckpt/fabind_plus_best_ckpt.bin}"
NUM_THREADS="${NUM_THREADS:-0}"

# Check if protein preprocessing has been run
if [[ ! -f "$DATA_DIR/prots.json" ]] || [[ ! -d "$FABIND_WORK_DIR/repr_files" ]]; then
    echo "ERROR: Protein data not found. Please run preprocess_proteins.sh first." >&2
    echo "  Expected: $DATA_DIR/prots.json" >&2
    echo "  Expected: $FABIND_WORK_DIR/repr_files/" >&2
    exit 1
fi

CHECKPOINTS=(
        "./checkpoints/value_1.ckpt"
        "./checkpoints/value_2.ckpt"
)

if [[ ! -f "$SMILES_CSV" ]]; then
    echo "ERROR: SMILES_CSV not found: $SMILES_CSV" >&2
    exit 1
fi

mkdir -p "$LIGAND_DATA_DIR" "$OUT_DIR"

export SMILES_CSV PROT_ID DATA_DIR FABIND_WORK_DIR LIGAND_DATA_DIR

python - <<'PY'
import csv
import json
import os

smiles_csv = os.environ["SMILES_CSV"]
prot_id = os.environ["PROT_ID"]
data_dir = os.environ["DATA_DIR"]
ligand_data_dir = os.environ["LIGAND_DATA_DIR"]
fabind_dir = os.environ["FABIND_WORK_DIR"]

os.makedirs(ligand_data_dir, exist_ok=True)

smiles_map = {}
id_list = []

with open(smiles_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

if not rows:
        raise SystemExit(f"No rows found in {smiles_csv}")

header = [h.strip().lower() for h in rows[0]]
has_header = "smiles" in header
smiles_col = header.index("smiles") if has_header else 0
data_rows = rows[1:] if has_header else rows

for idx, row in enumerate(data_rows):
        if not row:
                continue
        smiles = row[smiles_col].strip()
        if not smiles:
                continue
        ligand_id = f"L{idx:06d}"
        smiles_map[ligand_id] = smiles
        id_list.append(f"{prot_id}_{ligand_id}")

smiles_json = os.path.join(ligand_data_dir, "smiles.json")
id_json = os.path.join(ligand_data_dir, "id.json")

with open(smiles_json, "w", encoding="utf-8") as f:
        json.dump(smiles_map, f, indent=2)

with open(id_json, "w", encoding="utf-8") as f:
        json.dump(id_list, f, indent=2)

smiles_csv_out = os.path.join(ligand_data_dir, "smiles.csv")
data_csv_out = os.path.join(ligand_data_dir, "data.csv")

with open(smiles_csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "ligand_id"])
        for ligand_id, smiles in smiles_map.items():
                writer.writerow([smiles, ligand_id])

with open(data_csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Smiles", "prot_id", "ligand_id"])
        for ligand_id, smiles in smiles_map.items():
                writer.writerow([smiles, prot_id, ligand_id])

print(f"Wrote {len(smiles_map)} ligands to {smiles_json}")
print(f"Wrote {len(id_list)} ids to {id_json}")
print(f"Wrote FABind+ CSVs to {ligand_data_dir}")
PY

# Run FABind+ preprocessing and docking using bash scripts in fabind environment
cd FABind_plus

echo "======  preprocess molecules  ======"
conda run -n fabind_h100 bash -c "
    smiles_csv='$LIGAND_DATA_DIR/smiles.csv'
    num_threads='$NUM_THREADS'
    save_pt_dir='$FABIND_WORK_DIR/repr_files'
    
    python ./fabind/inference_preprocess_mol_confs.py --index_csv \${smiles_csv} --save_mols_dir \${save_pt_dir} --num_threads \${num_threads} --resume
"

echo "======  fabind inference  ======"
conda run -n fabind_h100 bash -c "
    index_csv='$LIGAND_DATA_DIR/data.csv'
    save_pt_dir='$FABIND_WORK_DIR/repr_files'
    ckpt_path='$FABIND_CKPT'
    output_dir='$LIGAND_DATA_DIR'
    
    python ./fabind/inference_regression_fabind.py \
        --ckpt \${ckpt_path} \
        --batch_size 4 \
        --post-optim \
        --write-mol-to-file \
        --sdf-output-path-post-optim \${output_dir} \
        --index-csv \${index_csv} \
        --preprocess-dir \${save_pt_dir}
"

cd "$ROOT_DIR"

echo "======  representation extraction  ======"

mkdir -p "$LIGAND_DATA_DIR/repr"

conda run -n fabind_h100 python src/affinity/data/repr/torchdrug.py \
    --input_json "$LIGAND_DATA_DIR/smiles.json" \
    --output_lmdb "$LIGAND_DATA_DIR/repr/torchdrug.lmdb" \
    --n_jobs -1

echo "======  flashaffinity inference  ======"

conda run -n flashaffinity torchrun --nproc_per_node=1 --rdzv_endpoint="localhost:29501" ./scripts/predict.py \
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
    --affinity_checkpoint "${CHECKPOINTS[@]}"
