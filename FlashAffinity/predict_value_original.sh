#!/bin/bash


CHECKPOINTS=(
        "./checkpoints/value_1.ckpt"
        "./checkpoints/value_2.ckpt"
)

torchrun --nproc_per_node=1 --rdzv_endpoint="localhost:29501" ./scripts/predict.py \
    --data "$DATA_DIR/id.json" \
    --task value \
    --structure "$DATA_DIR/pdb" \
    --structure_type pdb \
    --ligand "$DATA_DIR/ligand_sdf.lmdb" \
    --ligand_type sdf \
    --protein_repr "$DATA_DIR/repr/esm3.lmdb" \
    --ligand_repr "$DATA_DIR/repr/torchdrug.lmdb" \
    --distance_threshold 20.0 \
    --out_dir "$OUT_DIR" \
    --devices 1 \
    --affinity_checkpoint "${CHECKPOINTS[@]}"
