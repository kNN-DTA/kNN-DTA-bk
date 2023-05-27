python evaluate_kNN.py \
    --task dti_separate_add_mask_token \
    --datastore-path $dstore_path \
    --result-file-path $result_path \
    --prediction-mode combine \
    --dataset $dataset \
    --T $T --k $k --l $l \
    --T-0 $T_mol --k-0 $k_mol --knn-embedding-weight-0 $l_mol \
    --T-1 $T_pro --k-1 $k_pro --knn-embedding-weight-1 $l_pro \
    --batch-size 32 \
    --valid-subset test \
    --criterion dti_separate_knn_cls_eval_no_cross_attn \
    --path $ckpt_path \
    $data_path