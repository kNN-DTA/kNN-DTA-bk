ulimit -c unlimited

[ -z "${data_path}" ] && data_path='./data-bin/BindingDB_Ki'
[ -z "${save_path}" ] && save_path='./checkpoints'
[ -z "${save_prefix}" ] && save_prefix='pretrain'
[ -z "${encoder_layers}" ] && encoder_layers=16
[ -z "${total_steps}" ] && total_steps=35600  # 100 epochs through IMDB for bsz 32
[ -z "${warmup_steps}" ] && warmup_steps=1780      # 5 epochs of the number of updates
[ -z "${dataset_name}" ] && dataset_name="BindingDB_Ki"
[ -z "${pretrained_mol_ckpt}" ] && pretrained_mol_ckpt="./pretrained_ckpt/pubchem_L12.pt"
[ -z "${pretrained_pro_ckpt}" ] && pretrained_pro_ckpt="./pretrained_ckpt/pfam_L12.pt"

[ -z "${seed}"] && seed=1
[ -z "${dropout}" ] && dropout=0.1
[ -z "${attn_dropout}" ] && attn_dropout=0.1
[ -z "${lr}" ] && lr=1e-04
[ -z "${batch_size}" ] && batch_size=32
[ -z "${update_freq}" ] && update_freq=1
[ -z "${weight_decay}" ] && weight_decay=0.0
[ -z "${clip_norm}" ] && clip_norm=1.0

[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086
[ -z "${OMPI_COMM_WORLD_SIZE}" ] && OMPI_COMM_WORLD_SIZE=1


echo -e "\n\n"
echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "OMPI_COMM_WORLD_RANK: ${OMPI_COMM_WORLD_RANK}"
echo "OMPI_COMM_WORLD_SIZE: ${OMPI_COMM_WORLD_SIZE}"

if [[ -z "${OMPI_COMM_WORLD_SIZE}" ]]
then
  ddp_options=""
else
  if (( $OMPI_COMM_WORLD_SIZE == 1))
  then
	ddp_options=""
  else
    ddp_options="--nnodes=$OMPI_COMM_WORLD_SIZE --node_rank=$OMPI_COMM_WORLD_RANK --master_addr=$MASTER_ADDR"
  fi
fi
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="

hyperparams=feature_based-$dataset_name-lr-$lr-tsteps-$total_steps-wsteps-$warmup_steps-BS$((batch_size*n_gpu*OMPI_COMM_WORLD_SIZE*update_freq))-SEED$seed-CLIP$clip_norm-dp$dropout-attn_dp$attn_dropout-wd$weight_decay
save_dir=$save_path/$save_prefix-$hyperparams
tsb_dir=$save_dir/tsb
mkdir -p $save_dir

echo -e "\n\n"
echo "=====================================ARGS======================================"
echo "arg0: $0"
echo "seed: ${seed}"
echo "batch_size: $((batch_size*n_gpu*OMPI_COMM_WORLD_SIZE*update_freq))"
echo "lr: ${lr}"
echo "warmup_steps: ${warmup_steps}"
echo "total_steps: ${total_steps}"
echo "clip_norm: ${clip_norm}"
echo "update_freq: ${update_freq}"
echo "dropout: ${dropout}"
echo "attn_dropout: ${attn_dropout}"
echo "weight_decay: ${weight_decay}"
echo "save_dir: ${save_dir}"
echo "tsb_dir: ${tsb_dir}"
echo "data_dir: ${data_path}"
echo "dataset_name: ${dataset_name}"
echo "encoder_layers": ${encoder_layers}
echo "pretrained_mol_ckpt: ${pretrained_mol_ckpt}"
echo "pretrained_pro_ckpt: ${pretrained_pro_ckpt}"
echo "==============================================================================="

# ENV
echo -e "\n\n"
echo "======================================ENV======================================"
echo 'Environment'
ulimit -c unlimited;
echo '\n\nhostname'
hostname
echo '\n\nnvidia-smi'
nvidia-smi
echo '\n\nls -alh'
ls -alh
echo -e '\n\nls ~ -alh'
ls ~ -alh
echo "torch version"
python -c "import torch; print(torch.__version__)"
echo "==============================================================================="

export NCCL_ASYNC_ERROR_HADNLING=1
export OMP_NUM_THREADS=1


python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $ddp_options \
    $(which fairseq-train) \
    --task dti_separate_add_mask_token $data_path \
    --seed $seed \
    --num-classes 1 --init-token 0 \
    --max-positions-molecule 512 --max-positions-protein 1024 \
    --save-dir $save_dir \
    --encoder-layers $encoder_layers \
    --criterion dti_separate --regression-target \
    --batch-size $batch_size --update-freq $update_freq --required-batch-size-multiple 1 \
    --optimizer adam --weight-decay $weight_decay --adam-betas '(0.9,0.98)' --adam-eps 1e-06 \
    --lr-scheduler polynomial_decay --lr $lr --warmup-updates $warmup_steps --total-num-update $total_steps \
    --clip-norm $clip_norm --max-update $total_steps \
    --arch dti_knn_from_pretrained_roberta_no_cross_attn_1 --dropout $dropout --attention-dropout $attn_dropout \
    --skip-invalid-size-inputs-valid-test \
    --fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
    --shorten-method truncate \
    --tensorboard-logdir $tsb_dir \
    --pretrained-molecule-roberta-checkpoint $pretrained_mol_ckpt \
    --pretrained-protein-roberta-checkpoint $pretrained_pro_ckpt \
    --find-unused-parameters | tee -a $save_dir/training.log
