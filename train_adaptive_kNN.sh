set -x
set -e

DATASET="BindingDB_Ki"
TOTAL_NUM_UPDATES=3170
K=8
K_MOL=8
K_PRO=8
LR=1e-3

UPDATE_FREQ=1
BATCH_SIZE=32  
SEED=1
WARMUP_RATE=20
META_HIDDEN=32
META_HIDDEN_MOL=32
META_HIDDEN_PRO=32

PRETRAINED_MOLECULE_PROTEIN_ROBERTA_CKPT="./checkpoints/Ki_ckpt/checkpoint_best.pt"
DATASTORE_PATH="./checkpoints/Ki_ckpt/dstore"

# 32 9-10 GB
# 64 14-15 GB
# 128 20-21 GB
# 256 35-36 GB

TRAIN_SUBSET="valid"

while [[ $# -gt 0 ]]; do
    key=$1
    case $key in
    -d | --dataset)
        DATASET=$2
        shift 2
        ;;
    --total-num-updates)
        TOTAL_NUM_UPDATES=$2
        shift 2
        ;;
    -u | --update-freq)
        UPDATE_FREQ=$2
        shift 2
        ;;
    --train-subset)
        TRAIN_SUBSET=$2
        shift 2
        ;;
    --k)
        K=$2
        shift 2
        ;;
    --k-mol)
        K_MOL=$2
        shift 2
        ;;
    --k-pro)
        K_PRO=$2
        shift 2
        ;;
    --lr)
        LR=$2
        shift 2
        ;;
    --meta-hidden)
        META_HIDDEN=$2
        shift 2
        ;;
    --meta-hidden-mol)
        META_HIDDEN_MOL=$2
        shift 2
        ;;
    --meta-hidden-pro)
        META_HIDDEN_PRO=$2
        shift 2
        ;;
    --batch-size)
        BATCH_SIZE=$2
        shift 2
        ;;
    --seed)
        SEED=$2
        shift 2
        ;;
    --warmup-rate)
        WARMUP_RATE=$2
        shift 2
        ;;
    --datastore-path)
        DATASTORE_PATH=$2
        shift 2
        ;;
    --pretrained-molecule-protein-roberta-ckpt)
        PRETRAINED_MOLECULE_PROTEIN_ROBERTA_CKPT=$2
        shift 2
        ;;
    *)
        POSITIONAL+=("$1")
        shift
        ;;
    esac
done


WARMUP_UPDATES=$[TOTAL_NUM_UPDATES/WARMUP_RATE]      # 5% epochs of the number of updates

EXP_NAME=${DATASET}_lr${LR}_bsz${BATCH_SIZE}_total_updates${TOTAL_NUM_UPDATES}_warmup_updates${WARMUP_UPDATES}_${TRAIN_SUBSET}_k${K}_k_mol${K_MOL}_k_pro${K_PRO}_mh${META_HIDDEN}_mhmol${META_HIDDEN_MOL}_mhpro${META_HIDDEN_PRO}_v3_relu_seed${SEED}_1109

DTI_BIN=./data-bin/${DATASET}

SAVE_PATH=./checkpoints/adaptive_knn_training/${EXP_NAME}
mkdir -p $SAVE_PATH

TENSORBOARD_PATH=./$SAVE_PATH/tsb

# For distributed training
# python -m torch.distributed.launch --nproc_per_node=${GPU_PER_NODE_COUNT} --node_rank=${NODE_RANK} --nnodes=${NODE_COUNT} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
python $(which fairseq-train) \
    --seed $SEED \
    --task dti_separate_add_mask_token_no_register_class $DTI_BIN \
    --train-subset $TRAIN_SUBSET --valid-subset valid \
    --num-classes 1 --init-token 0 \
    --max-positions-molecule 512 --max-positions-protein 1024 \
    --save-dir $SAVE_PATH \
    --encoder-layers 16 \
    --criterion dti_separate --regression-target \
    --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ --required-batch-size-multiple 1 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-08 \
    --lr-scheduler polynomial_decay --lr $LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_NUM_UPDATES \
    --clip-norm 1.0 --max-update $TOTAL_NUM_UPDATES \
    --validate-interval 5 --save-interval 5 \
    --disable-validation \
    --arch dti_knn_training_adaptive_v3_relu \
    --skip-invalid-size-inputs-valid-test \
    --shorten-method truncate \
    --pretrained-molecule-protein-roberta-checkpoint $PRETRAINED_MOLECULE_PROTEIN_ROBERTA_CKPT \
    --fix-classification-head \
    --datastore-path $DATASTORE_PATH \
    --knn-sim-func do_not_recomp_l2 \
    --knn-lambda-type trainable --knn-temperature-type fix \
    --knn-k-type trainable --k-lambda-net-hid-size $META_HIDDEN --k-lambda-net-hid-size-mol $META_HIDDEN_MOL --k-lambda-net-hid-size-pro $META_HIDDEN_PRO --k-lambda-net-dropout-rate 0.0 \
    --k $K \
    --k-mol $K_MOL \
    --k-pro $K_PRO \
    --model-eval \
    --apply-layer-norm \
    --find-unused-parameters \
    --azureml-logging \
    --tensorboard-logdir $TENSORBOARD_PATH | tee -a ${SAVE_PATH}/training.log


