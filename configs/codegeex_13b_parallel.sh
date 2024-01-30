# CodeGeeX-13B parallel configuration
# Parallel checkpoints are named under the format "mp_rank_0{i}_model_states.pt", where i is the rank, start from 0.

CHECKPOINT_PATH="/data0/csw/CodeGeeX/scripts/mp2_parallel_weights/"

MODEL_ARGS="--num-layers 39 \
            --hidden-size 5120 \
            --num-attention-heads 40 \
            --max-position-embeddings 2048 \
            --attention-softmax-in-fp32 \
            --load "$CHECKPOINT_PATH" \
            --layernorm-epsilon 1e-5 \
            --fp16 \
            --ws-encoding-start-id 10 \
            --ws-encoding-length 10 \
            --make-vocab-size-divisible-by 52224 \
            --seq-length 2048"