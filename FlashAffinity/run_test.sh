#!/bin/bash
set -euo pipefail

# Accept run name as first argument, default to "test"
RUN_NAME="${1:-test}"

echo "Running with run name: $RUN_NAME"

# Ensure HF token is available for model downloads
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f ~/.cache/huggingface/token ]]; then
    export HF_TOKEN=$(cat ~/.cache/huggingface/token)
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "Warning: HF_TOKEN not set. ESM3 model access may fail."
    echo "Run: conda run -n esm3 huggingface-cli login"
fi

# Change to script directory
cd "$(dirname "${BASH_SOURCE[0]}")"

# Set up paths based on run name
export DATA_DIR="$PWD/data/$RUN_NAME"
export OUT_DIR="$PWD/value/$RUN_NAME"

# Protein preprocessing
PROT_FASTA=/disk1/jyang4/repos/SGPO-Genesis/boltz_templates/batch_inputs/2VT4.fasta \
PROT_CIF=/disk1/jyang4/repos/SGPO-Genesis/boltz_templates/batch_inputs/2VT4.cif \
PROT_CHAIN=A \
PROT_ID=2VT4 \
bash preprocess_proteins.sh

# Ligand prediction
SMILES_CSV=/disk1/jyang4/repos/genmol/FlashAffinity/data/my_smiles.csv \
PROT_ID=2VT4 \
bash predict_value.sh