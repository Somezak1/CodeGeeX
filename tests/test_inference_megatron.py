import copy
import time
import torch
import numpy as np

from codegeex.megatron import get_tokenizer, get_args, print_rank_0
from codegeex.megatron.initialize import initialize_megatron
from codegeex.megatron.model import CodeGeeXModel
from codegeex.megatron.code_generation_utils import get_token_stream
from codegeex.quantization import quantize
from codegeex.megatron.training import get_model
from codegeex.megatron.checkpointing import load_checkpoint

torch.set_printoptions(precision=8)


def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    
    print_rank_0("Building CodeGeeX model ...")
    model = CodeGeeXModel(num_tokentypes=0,
                          parallel_output=False)
    # 训练时model定义为CodeGeeXModel(num_tokentypes=0, parallel_output=True)

    return model


def add_code_generation_args(parser):
    """Code generation arguments."""
    group = parser.add_argument_group(title="code generation")

    group.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature.",
    )
    group.add_argument(
        "--greedy",
        action="store_true",
        default=False,
        help="Use greedy sampling.",
    )
    group.add_argument(
        "--top-p",
        type=float,
        default=0.0,
        help="Top p sampling.",
    )
    group.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top k sampling.",
    )
    group.add_argument(
        "--out-seq-length",
        type=int,
        default=2048,
        help="Size of the output generated text.",
    )
    group.add_argument(
        "--recompute",
        action="store_true",
        help="During generation recompute all attention "
             "instead of using previously computed keys/values.",
    )
    group.add_argument(
        "--ws-encoding-start-id",
        type=int,
        default=10,
        help="Start id for whitespace encoding",
    )
    group.add_argument(
        "--ws-encoding-length",
        type=int,
        default=10,
        help="Length of whitespace encoding",
    )
    group.add_argument(
        "--n-generation",
        type=int,
        default=10,
    )
    group.add_argument(
        "--eos-id",
        type=int,
        default=50256,
    )
    group.add_argument(
        "--prompt-file",
        type=str,
        default="./test_prompt.txt",
    )
    group.add_argument(
        "--perf-file",
        type=str,
        default="./perf_out.txt",
    )
    group.add_argument(
        "--perf-trace",
        type=str,
        default="./perf_out.txt",
    )
    group.add_argument(
        "--use-torch-profile",
        action="store_true",
    )
    group.add_argument(
        "--ln-fp32",
        action="store_true",
    )
    group.add_argument(
        '--bad-ids',
        nargs="*",
        type=int,
        default=None,
        help='Identify the type of programming language to generate',
    )
    group.add_argument(
        "--quantize",
        action="store_true",
    )

    return parser


def main():
    initialize_megatron(
        extra_args_provider=add_code_generation_args,
        args_defaults={
            'no_load_rng': True,
            'no_load_optim': True,
        }
    )
    # 训练时用的initialize_megatron(extra_args_provider=None, args_defaults={"tokenizer_type": "GPT2BPETokenizer"})

    args = get_args()
    set_random_seed(args.seed)
    # args.seed: 1234

    print_rank_0("Loading tokenizer ...")
    tokenizer = get_tokenizer()

    print_rank_0("Loading state dict ...")

    model = get_model(model_provider)
    # 训练时获取模型也是用的get_model函数, 不过传入的是另一套model_provider
    # args.load: /data0/csw/CodeGeeX/scripts/mp2_parallel_weights/
    if args.load is not None:
        _ = load_checkpoint(model, None, None)
    
    assert len(model) == 1, "Above condition should have caught this"
    
    model = model[0]
    model.eval()
    # args.fp16: True
    # args.ln_fp16: True
    if args.fp16 and args.ln_fp16:
        model.half()
    if args.quantize:
        model = quantize(model, weight_bit_width=8, backend="megatron")

    with open(args.prompt_file, "r") as f:
        prompt = f.readlines()
        prompt = "".join(prompt)

    # prompt:
    # "code translation
    # Java:
    # public class Solution {
    #     public static boolean hasCloseElements(int[] nums, int threshold) {
    #         for (int i = 0; i < nums.length - 1; i++) {
    #             for (int j = i + 1; j < nums.length; j++) {
    #                 if (Math.abs(nums[i] - nums[j]) < threshold) {
    #                     return true;
    #                 }
    #             }
    #         }
    #         return false;
    #     }
    # }
    # Python:"

    times = {}
    out_seq_lengths = [args.out_seq_length]
    # args.out_seq_length: 1024
    micro_batch_size = args.micro_batch_size
    # args.micro_batch_size: 1
    for out_seq_length in out_seq_lengths:        
        print_rank_0(f"Generating with out_seq_len {out_seq_length}...")
        
        times[out_seq_length] = []
        # args.n_generation: 1
        for prompt in [prompt] * args.n_generation:
            t0 = time.perf_counter()
            tokens = tokenizer.tokenize(prompt)
            # tokens:
            # [8189, 11059, 198, 29584, 25, 198, 11377, 1398, 28186, 1391, 198, 50268, 11377, 9037, 25131, 468, 26125,
            # 36, 3639, 7, 600, 21737, 997, 82, 11, 493, 11387, 8, 1391, 198, 50272, 1640, 357, 600, 1312, 796, 657,
            # 26, 1312, 1279, 997, 82, 13, 13664, 532, 352, 26, 1312, 29577, 1391, 198, 50274, 50266, 1640, 357, 600,
            # 474, 796, 1312, 1343, 352, 26, 474, 1279, 997, 82, 13, 13664, 26, 474, 29577, 1391, 198, 50274, 50270,
            # 361, 357, 37372, 13, 8937, 7, 77, 5700, 58, 72, 60, 532, 997, 82, 58, 73, 12962, 1279, 11387, 8, 1391,
            # 198, 50274, 50274, 7783, 2081, 26, 198, 50274, 50270, 92, 198, 50274, 50266, 92, 198, 50272, 92, 198,
            # 50272, 7783, 3991, 26, 198, 50268, 92, 198, 92, 198, 37906, 25, 198]
            print_rank_0(tokens)
            print_rank_0("Current prompt:")
            print_rank_0(prompt)
            n_token_prompt = len(tokens)
            print_rank_0(f"N_token_prompt:{n_token_prompt}")
            token_stream = get_token_stream(
                model,
                [copy.deepcopy(tokens) for _ in range(micro_batch_size)],
                micro_batch_size=micro_batch_size,
                topk=args.top_k,
                # args.top_k: 0
                topp=args.top_p,
                # args.top_p: 0.95
                temperature=args.temperature,
                # args.temperature: 0.8
            )
            is_finished = [False for _ in range(micro_batch_size)]
            for i, generated in enumerate(token_stream):
                generated_tokens = generated[0]
                for j in range(micro_batch_size):
                    if is_finished[j]:
                        continue
                    if generated_tokens[j].cpu().numpy()[-1] == tokenizer.eod or len(
                            generated_tokens[j]) >= out_seq_length:
                        # tokenizer.eod: 50256
                        is_finished[j] = True
                        generated_tokens_ = generated_tokens[j].cpu().numpy().tolist()
                        generated_code = tokenizer.detokenize(generated_tokens_[n_token_prompt:])
                        t1 = time.perf_counter()
                        print_rank_0(f"Total generation time: {t1 - t0}, # Tokens: {len(generated_tokens_) - n_token_prompt}")
                        print_rank_0(f"{(t1 - t0) / (len(generated_tokens_) - n_token_prompt)}s/token")
                        times[out_seq_length].append(t1 - t0)
                        print_rank_0("================================= Generated code:")
                        print_rank_0(generated_code)
                        t0 = time.perf_counter()
                    
                    if all(is_finished):
                        break

    print_rank_0(times)
    for out_seq_length in times.keys():
        print_rank_0(f"{out_seq_length}, {np.mean(times[out_seq_length])}")
        
    print_rank_0("Generation finished.")

if __name__ == "__main__":
    main()
