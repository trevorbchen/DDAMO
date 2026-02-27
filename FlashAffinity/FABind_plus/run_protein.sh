pdb_file_dir=${PDB_FILE_DIR:-./examples/pdb}
save_pt_dir=${SAVE_PT_DIR:-./examples/repr_files}

echo "======  preprocess proteins  ======"
python ./fabind/inference_preprocess_protein.py --pdb_file_dir ${pdb_file_dir} --save_pt_dir ${save_pt_dir}
