# This script is used to test the inference of CodeGeeX.

GPU=$1
PROMPT_FILE=$2

SCRIPT_PATH=$(realpath "$0")
# SCRIPT_PATH: /home/icksys/csw/CodeGeeX/scripts/test_inference.sh
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
# SCRIPT_DIR: /home/icksys/csw/CodeGeeX/scripts/
MAIN_DIR=$(dirname "$SCRIPT_DIR")
# MAIN_DIR: /home/icksys/csw/CodeGeeX/
TOKENIZER_PATH="$MAIN_DIR/codegeex/tokenizer/"
# TOKENIZER_PATH: /home/icksys/csw/CodeGeeX/codegeex/tokenizer/

# import model configuration
source "$MAIN_DIR/configs/codegeex_13b.sh"

# export CUDA settings
if [ -z "$GPU" ]; then
  GPU=0
fi

export CUDA_HOME=/usr/local/cuda-11.7/
export CUDA_VISIBLE_DEVICES=$GPU

if [ -z "$PROMPT_FILE" ]; then
  PROMPT_FILE=$MAIN_DIR/tests/test_prompt.txt
fi

# remove --greedy if using sampling
CMD="python $MAIN_DIR/tests/test_inference.py \
        --prompt-file $PROMPT_FILE \
        --tokenizer-path $TOKENIZER_PATH \
        --micro-batch-size 1 \
        --out-seq-length 1024 \
        --temperature 0.8 \
        --top-p 0.95 \
        --top-k 0 \
        --greedy \
        $MODEL_ARGS"

# MODEL_ARGS="--num-layers 39 \
#             --hidden-size 5120 \
#             --num-attention-heads 40 \
#             --max-position-embeddings 2048 \
#             --attention-softmax-in-fp32 \
#             --load /home/icksys/csw/CodeGeeX/scripts/codegeex_13b.pt \
#             --layernorm-epsilon 1e-5 \
#             --fp16 \
#             --ws-encoding-start-id 10 \
#             --ws-encoding-length 10 \
#             --make-vocab-size-divisible-by 52224 \
#             --seq-length 2048"

echo "$CMD"
#eval "$CMD"
