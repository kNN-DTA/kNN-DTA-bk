python build_datastore.py \
    --task dti_separate_add_mask_token \
    --dataset $dataset \
    --datastore-path $dstore_path \
    --batch-size 32 \
    --valid-subset train \
    --criterion dti_separate_knn_build_datastore \
    --path $ckpt_path \
    $data_path