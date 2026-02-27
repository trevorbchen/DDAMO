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
PROT_ID="${PROT_ID:-2VT4}"
PROT_SEQ="${PROT_SEQ:-}"
PROT_FASTA="${PROT_FASTA:-}"
PROT_CIF="${PROT_CIF:-}"  # Optional: path to existing CIF file to use instead of running structure prediction
PROT_CHAIN="${PROT_CHAIN:-A}"  # Chain to use from CIF file
DATA_DIR="${DATA_DIR:-$ROOT_DIR/data/test}"
FABIND_WORK_DIR="${FABIND_WORK_DIR:-$ROOT_DIR/FABind_plus/working_dir}" #TODO make this a temporary directory?

# Read protein sequence from FASTA if not provided
if [[ -z "$PROT_SEQ" ]]; then
    if [[ -n "$PROT_FASTA" ]]; then
        PROT_SEQ="$(conda run -n fabind_h100 python - <<'PY'
import os
from Bio import SeqIO

fasta = os.environ.get('PROT_FASTA')
if not fasta:
        raise SystemExit("PROT_FASTA is empty")

records = list(SeqIO.parse(fasta, "fasta"))
if not records:
        raise SystemExit(f"No sequences found in {fasta}")

print(str(records[0].seq).strip())
PY
        )"
    else
        echo "ERROR: Provide PROT_SEQ or PROT_FASTA." >&2
        exit 1
    fi
fi

# Create directories
mkdir -p "$DATA_DIR" "$DATA_DIR/repr" "$DATA_DIR/pdb" "$FABIND_WORK_DIR" "$FABIND_WORK_DIR/pdb" "$FABIND_WORK_DIR/repr_files"

export PROT_ID PROT_SEQ DATA_DIR FABIND_WORK_DIR PROT_CIF PROT_CHAIN

# Create protein JSON file
conda run -n fabind_h100 python - <<'PY'
import json
import os

prot_id = os.environ["PROT_ID"]
prot_seq = os.environ["PROT_SEQ"]
data_dir = os.environ["DATA_DIR"]

os.makedirs(data_dir, exist_ok=True)

prots_json = os.path.join(data_dir, "prots.json")

with open(prots_json, "w", encoding="utf-8") as f:
        json.dump({prot_id: prot_seq}, f, indent=2)

print(f"Wrote protein info to {prots_json}")
PY

# Structure preparation: use provided CIF or run structure prediction
if [[ -n "$PROT_CIF" ]]; then
    echo "Using provided CIF file: $PROT_CIF (chain $PROT_CHAIN)"
    
    if [[ ! -f "$PROT_CIF" ]]; then
        echo "ERROR: PROT_CIF file not found: $PROT_CIF" >&2
        exit 1
    fi
    
    # Convert CIF to PDB using Python
    conda run -n fabind_h100 python - <<PY
import os
from Bio.PDB import MMCIFParser, PDBIO, Select

class ChainSelect(Select):
    def __init__(self, chain_id):
        self.chain_id = chain_id
    
    def accept_chain(self, chain):
        return chain.id == self.chain_id

cif_file = os.environ["PROT_CIF"]
prot_id = os.environ["PROT_ID"]
chain_id = os.environ["PROT_CHAIN"]
output_dir = os.environ["DATA_DIR"] + "/pdb"

parser = MMCIFParser(QUIET=True)
structure = parser.get_structure(prot_id, cif_file)

io = PDBIO()
io.set_structure(structure)

output_pdb = os.path.join(output_dir, f"{prot_id}.pdb")
io.save(output_pdb, ChainSelect(chain_id))

print(f"Converted {cif_file} (chain {chain_id}) to {output_pdb}")
PY
else
    echo "Running structure prediction with fold.py"
    conda run -n fabind_h100 python src/affinity/data/fold.py \
        --prots_json_path "$DATA_DIR/prots.json" \
        --output_pdb_dir "$DATA_DIR/pdb" \
        --work_dir "$DATA_DIR/boltz_work"
fi

# Copy PDB files to FABind work directory
cp -a "$DATA_DIR/pdb/." "$FABIND_WORK_DIR/pdb/"

# Run FABind+ protein preprocessing
cd FABind_plus

echo "======  preprocess proteins  ======"
PROTEIN_LOG="$FABIND_WORK_DIR/repr_files/protein_preprocessing.log"
echo "Logging protein preprocessing to: $PROTEIN_LOG"
conda run -n fabind_h100 bash -c "
    pdb_file_dir='$FABIND_WORK_DIR/pdb'
    save_pt_dir='$FABIND_WORK_DIR/repr_files'
    
    python ./fabind/inference_preprocess_protein_optimized.py --pdb_file_dir \${pdb_file_dir} --save_pt_dir \${save_pt_dir} 2>&1 | tee -a '$PROTEIN_LOG'
"

cd "$ROOT_DIR"

# Extract protein representations
echo "======  extract protein representations  ======"
ESM_CACHE_DIR="$FABIND_WORK_DIR/../esm_cache" conda run -n esm3 python src/affinity/data/repr/esm3.py \
    --input_json "$DATA_DIR/prots.json" \
    --output_lmdb "$DATA_DIR/repr/esm3.lmdb" \
    --device auto

echo "Protein preprocessing complete!"
echo "Data directory: $DATA_DIR"
echo "FABind work directory: $FABIND_WORK_DIR"