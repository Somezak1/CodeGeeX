SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

# ====== Environment ======
# - NCCL & IB
export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

HOSTFILE=${SCRIPT_DIR}/csw_hostfile
MASTER_IP=$(cat $HOSTFILE | head -n 1)
cat $HOSTFILE | awk '{print $1 " slots=8"}' > $SCRIPT_DIR/hostfile
echo "MASTER_IP=$MASTER_IP"

# ====== Parameters ======
DATA_PATH=${MAIN_DIR}/sft_data/my_data
CKPT_PATH=${SCRIPT_DIR}/mp4_parallel_weights/
DS_CONFIG=${SCRIPT_DIR}/ds_config.json
# - 13b
TP=4
PP=1
NLAYERS=39
HIDDEN=5120
NATTN_HEAD=40
EMBED_VOCAB=52224
GLOBAL_BATCH=4
MICRO_BATCH=2
NTRAIN_ITERS=25
EVAL_INT=10
SAVE_INT=10
TRIAL_TAG="13b-test"
# - trial
TRIAL_NAME="finetune-codegeex"
# - zero stage
ZERO_STAGE=2
# - logging & output
OUTPUT_DIR="${SCRIPT_DIR}/csw-$TRIAL_NAME-$TRIAL_TAG"
mkdir -p $OUTPUT_DIR

# Deepspeed config
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH,
  "steps_per_print": 5,
  "zero_optimization": {
    "stage": $ZERO_STAGE,
    "reduce_bucket_size": 50000000,
    "allgather_bucket_size": 50000000,
    "overlap_comm": true,
    "contiguous_gradients": false
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "wall_clock_breakdown" : true
}
EOT

ds_args=""
ds_args=" --deepspeed ${ds_args}"
ds_args=" --no-pipeline-parallel ${ds_args}"
ds_args=" --deepspeed_config=$DS_CONFIG ${ds_args}"
ds_args=" --zero-stage=$ZERO_STAGE ${ds_args}"
ds_args=" --deepspeed-activation-checkpointing ${ds_args}"

echo "Launching deepspeed"
CUDA_VISIBLE_DEVICES=4,5,6,7 deepspeed \
    --hostfile hostfile \
    --master_addr $MASTER_IP \
    --master_port 29501 \
    $MAIN_DIR/codegeex/megatron/tools/finetune_codegeex.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --no-pipeline-parallel \
    --num-layers $NLAYERS \
    --hidden-size $HIDDEN \
    --make-vocab-size-divisible-by $EMBED_VOCAB \
    --num-attention-heads $NATTN_HEAD \
    --seq-length 128 \
    --loss-scale 12 \
    --max-position-embeddings 2048 \
    --micro-batch-size $MICRO_BATCH \
    --global-batch-size $GLOBAL_BATCH \
    --train-iters $NTRAIN_ITERS \
    --lr 1e-6 \
    --min-lr 1e-7 \
    --lr-decay-iters 100000 \
    --lr-decay-style cosine \
    --lr-warmup-iters 1000 \
    --log-interval 1 \
    --eval-iters 10 \
    --eval-interval $EVAL_INT \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --vocab-file $MAIN_DIR/codegeex/tokenizer/vocab.json \
    --merge-file $MAIN_DIR/codegeex/tokenizer/merges.txt \
    --save-interval $SAVE_INT \
    --save $OUTPUT_DIR \
    --load $OUTPUT_DIR \
    --load-state $CKPT_PATH \
    --split 98,2,0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --fp16 \
    --ln-fp16 \
    --attention-softmax-in-fp32 \
    --checkpoint-activations \
    --override-lr-scheduler \
    $ds_args |& tee ${OUTPUT_DIR}/$NOW.log