smiles_csv=${SMILES_CSV:-./examples/smiles.csv}
num_threads=${NUM_THREADS:-0}
save_pt_dir=${SAVE_PT_DIR:-./examples/repr_files}

echo "======  preprocess molecules  ======"
python ./fabind/inference_preprocess_mol_confs.py --index_csv ${smiles_csv} --save_mols_dir ${save_pt_dir} --num_threads ${num_threads} --resume
