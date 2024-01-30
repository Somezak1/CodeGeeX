import copy

from typing import *
from codegeex.tokenizer import CodeGeeXTokenizer
from codegeex.torch.inference import get_token_stream


def get_model(
    backend: str = "megatron",
    quantized: bool = False,
):
    pass


def generate(
    model, 
    tokenizer: CodeGeeXTokenizer, 
    prompt: str, 
    out_seq_length: int,
    # out_seq_length: 1024
    seq_length: int = 2048,
    # seq_length: 2048
    top_k: int = 0,
    # top_k: 0
    top_p: float = 1.0,
    # top_p: 0.95
    temperature: float = 1.0,
    # temperature: 0.8
    micro_batch_size: int = 1,
    # micro_batch_size: 1
    backend: str = "megatron",
    # backend: "megatron"
    greedy: bool = False,
    # greedy: False
    verbose: bool = False,
    # verbose: True
):
    tokens = tokenizer.encode_code(prompt)
    # tokens:
    # [8189, 11059, 198, 29584, 25, 198, 11377, 1398, 28186, 1391, 198, 50268, 11377, 9037, 25131, 468, 26125, 36,
    #  3639, 7, 600, 21737, 997, 82, 11, 493, 11387, 8, 1391, 198, 50272, 1640, 357, 600, 1312, 796, 657, 26, 1312,
    #  1279, 997, 82, 13, 13664, 532, 352, 26, 1312, 29577, 1391, 198, 50274, 50266, 1640, 357, 600, 474, 796, 1312,
    #  1343, 352, 26, 474, 1279, 997, 82, 13, 13664, 26, 474, 29577, 1391, 198, 50274, 50270, 361, 357, 37372, 13,
    #  8937, 7, 77, 5700, 58, 72, 60, 532, 997, 82, 58, 73, 12962, 1279, 11387, 8, 1391, 198, 50274, 50274, 7783, 2081,
    #  26, 198, 50274, 50270, 92, 198, 50274, 50266, 92, 198, 50272, 92, 198, 50272, 7783, 3991, 26, 198, 50268, 92,
    #  198, 92, 198, 37906, 25]
    n_token_prompt = len(tokens)
    # n_token_prompt: 126

    if verbose:
        print(f"Current prompt:\n{prompt}")
        print("N_token_prompt:", n_token_prompt)
    
    generated_codes = []
    if backend == "megatron":
        token_stream = get_token_stream(
            model,
            tokenizer,
            seq_length,
            out_seq_length,
            [copy.deepcopy(tokens) for _ in range(micro_batch_size)],
            micro_batch_size=micro_batch_size,
            topk=top_k,
            topp=top_p,
            temperature=temperature,
            greedy=greedy,
        )
        is_finished = [False for _ in range(micro_batch_size)]
        for i, generated in enumerate(token_stream):
            generated_tokens = generated[0]
            for j in range(micro_batch_size):
                if is_finished[j]:
                    continue
                
                if generated_tokens[j].cpu().numpy()[-1] == tokenizer.eos_token_id or len(generated_tokens[j]) >= out_seq_length:
                    is_finished[j] = True
                    generated_tokens_ = generated_tokens[j].cpu().numpy().tolist()
                    # generated_tokens_[n_token_prompt:]:
                    # [198, 50268, 4299, 468, 62, 19836, 62, 68, 3639, 7, 944, 11,
                    # 997, 82, 11, 11387, 2599, 198, 50272, 1640, 1312, 287, 2837, 7, 11925, 7, 77, 5700, 13219, 16,
                    # 2599, 198, 50274, 50266, 1640, 474, 287, 2837, 7, 72, 10, 16, 11, 18896, 7, 77, 5700, 8, 2599,
                    # 198, 50274, 50270, 361, 2352, 7, 77, 5700, 58, 72, 45297, 77, 5700, 58, 73, 12962, 1279, 11387,
                    # 25, 198, 50274, 50274, 7783, 6407, 198, 50272, 7783, 10352, 198, 198, 29584, 25, 198, 50268,
                    # 4299, 468, 26125, 36, 3639, 7, 944, 11, 997, 82, 11, 11387, 2599, 198, 50272, 7783, 23243, 7,
                    # 77, 5700, 8, 14512, 23243, 26933, 87, 12, 400, 10126, 329, 2124, 287, 997, 82, 12962, 198, 50256]
                    generated_code = tokenizer.decode_code(generated_tokens_[n_token_prompt:])
                    generated_code = "".join(generated_code)
                    # generated_code:
                    # '\n    def has_close_elements(self, nums, threshold):\n        for i in range(len(nums)-1):\n            for j in range(i+1, len(nums)):\n                if abs(nums[i]-nums[j]) < threshold:\n                    return True\n        return False\n\nJava:\n    def hasCloseElements(self, nums, threshold):\n        return sorted(nums)!= sorted([x-threshold for x in nums])\n<|endoftext|>'
                    generated_codes.append(generated_code)
                    if verbose:
                        print(f"\nGenerated code {i}:\n{generated_code}")
                    '''
                        def has_close_elements(self, nums, threshold):
                            for i in range(len(nums)-1):
                                for j in range(i+1, len(nums)):
                                    if abs(nums[i]-nums[j]) < threshold:
                                        return True
                            return False

                    Java:
                        def hasCloseElements(self, nums, threshold):
                            return sorted(nums)!= sorted([x-threshold for x in nums])
                    <|endoftext|>
                    '''
                    
                if all(is_finished):
                    break

    return generated_codes