"""Get model parallel partitions."""

import os
import torch
import argparse


def get_change_ckpt_args(parser):
    """Provide extra arguments required for merging."""
    group = parser.add_argument_group(title='Mindspore to megatron')
    group.add_argument(
        '--load-ckpt-path',
        type=str,
        required=True,
        help='path to load ".pt" checkpoint.',
    )
    group.add_argument(
        '--save-ckpt-path',
        type=str,
        required=True,
        help='dir to save converted checkpoints.',
    )
    group.add_argument(
        '--target-tensor-model-parallel-size',
        type=int,
        default=2,
        help='target tensor model parallel size',
    )
    
    return parser


def get_element_from_dict_by_path(d, path):
    """
    Get element from dictionary by path. If element is not present, recursively add empty dictionaries.
    Args:
        d (dict): the dictionary to get the element from
        path (list): the path to the element which is delimited by "."
    """
    path = path.split(".")
    for k in path:
        if k not in d:
            d[k] = {}
        d = d[k]
    return d


def main():
    # 注释代码
    # python /home/icksys/csw/CodeGeeX/codegeex/megatron/convert_ckpt_parallel.py       --load-ckpt-path codegeex_13b.pt       --save-ckpt-path mp2_parallel_weights_new       --tokenizer-path /home/icksys/csw/CodeGeeX/codegeex/tokenizer/       --target-tensor-model-parallel-size 2       --num-layers 39       --hidden-size 5120       --num-attention-heads 40       --max-position-embeddings 2048       --attention-softmax-in-fp32       --fp16       --micro-batch-size 1       --make-vocab-size-divisible-by 52224       --seq-length 2048
    parser = argparse.ArgumentParser()
    parser = get_change_ckpt_args(parser)
    args, _ = parser.parse_known_args()

    # args:
    # Namespace(
    #   load_ckpt_path='codegeex_13b.pt',
    #   save_ckpt_path='mp2_parallel_weights_new',
    #   target_tensor_model_parallel_size=2
    # )

    print(f"Load ckpt from {args.load_ckpt_path}...")
    state_dict = torch.load(args.load_ckpt_path, map_location="cpu")

    print(f"Spliting ckpt into {args.target_tensor_model_parallel_size} parts...")
    output_state_dict = []
    for i in range(args.target_tensor_model_parallel_size):
        output_state_dict.append({})

    print("Converting Embedding layers...")
    word_embeddings = state_dict['module']['language_model']['embedding']['word_embeddings']['weight']
    position_embeddings = state_dict['module']['language_model']['embedding']['position_embeddings']['weight']
    out_word_embeddings = torch.chunk(word_embeddings, args.target_tensor_model_parallel_size, dim=0)

    for i in range(args.target_tensor_model_parallel_size):
        pos_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "module.language_model.embedding.position_embeddings"
        )
        pos_emb_dict["weight"] = position_embeddings

        word_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "module.language_model.embedding.word_embeddings"
        )
        word_emb_dict["weight"] = out_word_embeddings[i].clone()

    print("Converting QueryEmbedding layers...")
    query_embeddings = state_dict['module']['language_model']['topQueryEmbedding']['top_query_embeddings']['weight']
    out_query_embeddings = torch.chunk(query_embeddings, args.target_tensor_model_parallel_size, dim=0)

    for i in range(args.target_tensor_model_parallel_size):
        query_emb_dict = get_element_from_dict_by_path(
            output_state_dict[i], "module.language_model.topQueryEmbedding.top_query_embeddings"
        )
        query_emb_dict["weight"] = out_query_embeddings[i].clone()

    print("Converting Transformer layers...")
    for layer_name in state_dict['module']['language_model']['transformer'].keys():
        params = state_dict['module']['language_model']['transformer'][layer_name]
        if "layernorm" in layer_name:
            # layers.i.input_layernorm.weight
            # layers.i.input_layernorm.bias
            # layers.i.post_attention_layernorm.weight
            # layers.i.post_attention_layernorm.bias
            # topQueryLayer.input_layernorm.weight
            # topQueryLayer.input_layernorm.bias
            # topQueryLayer.post_attention_layernorm.weight
            # topQueryLayer.post_attention_layernorm.bias
            # final_layernorm.weight
            # final_layernorm.bias
            pass
        elif "attention" in layer_name and "weight" in layer_name:
            if "dense" in layer_name:
                # layers.38.attention.dense.weight
                # topQueryLayer.attention.dense.weight
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=1)
            else:
                # layers.38.attention.query.weight
                # layers.38.attention.key.weight
                # layers.38.attention.value.weight
                # topQueryLayer.attention.query.weight
                # topQueryLayer.attention.key.weight
                # topQueryLayer.attention.value.weight
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)
        elif "weight" in layer_name and "dense" in layer_name:
            if "h_to_4h" in layer_name:
                # layers.38.mlp.dense_h_to_4h.weight
                # topQueryLayer.mlp.dense_h_to_4h.weight
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)
            else:
                # layers.38.mlp.dense_4h_to_h.weight
                # topQueryLayer.mlp.dense_4h_to_h.weight
                params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=1)
        elif "bias" in layer_name:
            if "dense" not in layer_name or "mlp" in layer_name:
                if "4h_to_h" in layer_name:
                    # layers.38.mlp.dense_4h_to_h.bias
                    # topQueryLayer.mlp.dense_4h_to_h.bias
                    pass
                else:
                    # layers.38.attention.query.bias
                    # layers.38.attention.key.bias
                    # layers.38.attention.value.bias
                    # layers.38.attention.dense.bias
                    # topQueryLayer.attention.query.bias
                    # topQueryLayer.attention.key.bias
                    # topQueryLayer.attention.value.bias
                    # topQueryLayer.attention.dense.bias
                    params = torch.chunk(params, args.target_tensor_model_parallel_size, dim=0)

        # 没遍历到的情况
        # layers.38.mlp.dense_h_to_4h.bias
        # topQueryLayer.mlp.dense_h_to_4h.bias
        for i in range(args.target_tensor_model_parallel_size):
            params_dict = get_element_from_dict_by_path(output_state_dict[i], "module.language_model.transformer")
            if type(params) is tuple:
                params_dict[layer_name] = params[i].clone()
            else:
                params_dict[layer_name] = params
    
    os.makedirs(args.save_ckpt_path, exist_ok=True)
    for rank in range(args.target_tensor_model_parallel_size):
        save_ckpt_path = os.path.join(args.save_ckpt_path, f"mp_rank_{rank:02d}_model_states.pt")
        torch.save(output_state_dict[rank], save_ckpt_path)
        print(f"Converted checkpoint saved in {save_ckpt_path}.")


if __name__ == '__main__':
    main()
