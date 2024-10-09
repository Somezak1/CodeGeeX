# This script is used to test the inference of CodeGeeX.

MP_SIZE=$1
PROMPT_FILE=$2

SCRIPT_PATH=$(realpath "$0")
# SCRIPT_PATH: /home/icksys/csw/CodeGeeX/scripts/test_inference_parallel.sh
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
# SCRIPT_DIR: /home/icksys/csw/CodeGeeX/scripts/
MAIN_DIR=$(dirname "$SCRIPT_DIR")
# MAIN_DIR: /home/icksys/csw/CodeGeeX/
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"
# TOKENIZER_PATH: /home/icksys/csw/CodeGeeX/codegeex/tokenizer/

if [ -z "$MP_SIZE" ]; then
  MP_SIZE=1
fi

if [ "$MP_SIZE" -eq 1 ]; then
  source "$MAIN_DIR/configs/codegeex_13b.sh"
  echo "Load config from $MAIN_DIR/configs/codegeex_13b.sh"
else
  source "$MAIN_DIR/configs/codegeex_13b_parallel.sh"
  echo "Load config from $MAIN_DIR/configs/codegeex_13b_parallel.sh"
fi

# export CUDA settings
export CUDA_HOME=/usr/local/cuda-11.7/
export CUDA_VISIBLE_DEVICES=0,1

if [ -z "$PROMPT_FILE" ]; then
  PROMPT_FILE=$MAIN_DIR/tests/test_prompt.txt
fi

# remove --greedy if using sampling
CMD="torchrun --nproc_per_node $MP_SIZE $MAIN_DIR/tests/test_inference_megatron.py \
        --tensor-model-parallel-size $MP_SIZE \
        --prompt-file $PROMPT_FILE \
        --tokenizer-path $TOKENIZER_PATH \
        --micro-batch-size 1 \
        --out-seq-length 1024 \
        --temperature 0.8 \
        --top-p 0.95 \
        --top-k 0 \
        --greedy \
        --use-cpu-initialization \
        --ln-fp16 \
        --n-generation 1\
        $MODEL_ARGS"

# MODEL_ARGS="--num-layers 39 \
#             --hidden-size 5120 \
#             --num-attention-heads 40 \
#             --max-position-embeddings 2048 \
#             --attention-softmax-in-fp32 \
#             --load /home/icksys/csw/CodeGeeX/scripts/mp2_parallel_weights/ \
#             --layernorm-epsilon 1e-5 \
#             --fp16 \
#             --ws-encoding-start-id 10 \
#             --ws-encoding-length 10 \
#             --make-vocab-size-divisible-by 52224 \
#             --seq-length 2048"

echo "$CMD"
eval "$CMD"
