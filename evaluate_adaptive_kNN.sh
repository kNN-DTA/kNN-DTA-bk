dataset="BindingDB_Ki"
data_path="./data-bin/BindingDB_Ki"
ckpt_path="./checkpoints/adaptive_knn_training/BindingDB_Ki_lr1e-03_bsz32_total_updates3170_warmup_updates158_valid_k8_k_mol8_k_pro8_mh32_mhmol32_mhpro32_v3_relu_seed22/checkpoint_last.pt"
result_path="./local_test.tsv"

python evaluate.py \
    --task dti_separate_add_mask_token_no_register_class \
    --batch-size 32 \
    --valid-subset test \
    --criterion dti_separate_eval \
    --path $ckpt_path \
    --output-fn $result_path \
    $data_path