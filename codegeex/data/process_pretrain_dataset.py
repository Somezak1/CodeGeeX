import os
import glob
import fire
import torch
import multiprocessing

from typing import *
from tqdm.auto import tqdm
from time import perf_counter
from black import format_str, FileMode

from codegeex.data.types import PromptDataset, PromptSample
from codegeex.data.processor import PromptDatasetProcessor
from codegeex.data.data_utils import stream_jsonl, LANGUAGE_TAG
from codegeex.megatron.data.indexed_dataset import make_mmap_builder
from codegeex.tokenizer import CodeGeeXTokenizer


def try_format_code(code: str):
    # Auto-correct to PEP8 format (Change tab to 4-whitespaces;
    # add whitespace around some special symbols;
    # reformat line length < 100, etc.)
    try:
        res = format_str(code, mode=FileMode(line_length=200))
    except Exception as e:
        res = code
        print(e)
        print("Wrong python format: {}".format(code))
    return res


def load_pretrain_dataset(dataset_path: Union[str, List[str]]) -> Dict:
    # ① dataset_path: "pt_data/my_data.jsonl"
    # ② dataset_path: "sft_data/my_data.jsonl"
    if type(dataset_path) is str:
        dataset_path = [dataset_path]
    
    for p in dataset_path:
        if not os.path.isdir(p):
            if p.endswith(".gz") or p.endswith(".jsonl"):
                print(f"loading from {p}")
                # ① p: "pt_data/my_data.jsonl"
                # ② p: "sft_data/my_data.jsonl"
                yield from stream_jsonl(p)
        else:
            p_list = glob.glob(p + "/*")
            for p_ in p_list:
                if p_.endswith(".gz") or p_.endswith(".jsonl"):
                    print(f"loading from {p_}")
                    yield from stream_jsonl(p_)
          
            
def process_sample(
    sample: Dict, 
    language: str=None, 
    mode: str="pretrain",
) -> Iterable[PromptSample]:
    if mode == "pretrain":
        prompt = ""
    else:
        prompt = sample["prompt"]
    # ① prompt: ''
    # ② prompt: 'Please translate below code into python:\nint main() {\n    int a = 0;\n    for (int i = 0; i < a; i++) {\n        std::cout << i << std::endl;\n    }\n    return 0;\n}\npython:\n'
    
    try:
        if language is not None and language in LANGUAGE_TAG.keys():
            code = LANGUAGE_TAG[language] + "\n" + sample["code"]
            # code: '# language: Python\ndef main():\n    a = 0\n    for i in range(a):\n        print(i)\n    return'
        else:
            code = sample["code"]
            # ② code:                  'def main():\n    a = 0\n    for i in range(a):\n        print(i)\n    return'
    except Exception as e:
        print(e)
        print("The key 'code' is missing in data. Aborted")
        exit(0)
        
    yield PromptSample(prompt, code)


def generate_prompt_samples(
    dataset: Iterable[Dict], 
    language: str = None,
    mode: str = "pretrain",
) -> PromptDataset:
    for sample in dataset:
        # sample is a dict
        yield from process_sample(sample, language, mode)


def main(
    tokenizer_path: str,
    # tokenizer_path: "/data0/csw/CodeGeeX/codegeex/tokenizer/"
    dataset_path: Union[str, List[str]],
    # ① dataset_path: "pt_data/my_data.jsonl"
    # ② dataset_path: "sft_data/my_data.jsonl"
    output_prefix: str,
    # ① output_prefix: "pt_data/my_data"
    # ② output_prefix: "sft_data/my_data"
    language: str = None,
    # ① language: "python"
    # ② language: None
    mode: str = "pretrain",
    # ① mode: "pretrain"
    # ② mode: "sft"
    discard_overlong: bool = False,
    sliding_stride: int = 200,
    num_workers: int = 32,
    seq_len: int = 2048,
    # seq_len: 128
):
    DATA_KEYS = ["input_ids", "attention_mask", "labels"]
    
    # create output dir
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    tokenizer = CodeGeeXTokenizer(tokenizer_path=tokenizer_path)
    pad_token_id = tokenizer.eos_token_id

    dataset = load_pretrain_dataset(dataset_path)
    # dataset 每次返回 jsonl 文件的一行, 格式是一个字典
    prompt_dataset = generate_prompt_samples(dataset, language=language, mode=mode)
    # prompt_dataset 每次返回一个对象, 其中包含了这个样本的 prompt 及 code 的信息

    if num_workers == 0:
        num_workers = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_workers)
    # 该进程池最多开启32个进程
    output_bin_files = {}
    output_idx_files = {}
    builders = {}

    for key in DATA_KEYS:
        output_bin_files[key] = "{}_{}.bin".format(output_prefix, key)
        output_idx_files[key] = "{}_{}.idx".format(output_prefix, key)
        # ① output_prefix: "pt_data/my_data"
        # ② output_prefix: "sft_data/my_data"
        builders[key] = make_mmap_builder(
            output_bin_files[key],
            vocab_size=None,  # magic number, should change it
        )

    # NOTE that we use seq_len + 1 instead of seq_len, since the input tokens will be shifted by one.
    processor = PromptDatasetProcessor(
        tokenize=tokenizer.encode_code, 
        pad_token=pad_token_id,
        max_seq_len=seq_len + 1, 
        discard_overlong=discard_overlong,
        sliding_stride=sliding_stride,
        eod_token=pad_token_id)
    
    processor.start_time = perf_counter()
    # imap_unordered 会同时开启32个异步进程进行 process_sample_strict 的执行, 不影响主进程继续运行, 只有在取结果时才会阻塞主进程
    doc_iter = pool.imap_unordered(processor.process_sample_strict,
                                   prompt_dataset, 
                                   chunksize=20)

    for doc_idx, docs in tqdm(enumerate(doc_iter, start=1)):
        processor.doc_processed += 1
        # docs: [{"input_ids": ..., "attention_mask": ..., "labels": ...}]
        for doc in docs:
            processor.doc_generated += 1
            for key in DATA_KEYS:
                builders[key].add_item(torch.IntTensor(doc[key]))

    for key in DATA_KEYS:
        builders[key].finalize(output_idx_files[key])


if __name__ == "__main__":
    # ① python codegeex/data/process_pretrain_dataset.py  --dataset_path  pt_data/my_data.jsonl  --tokenizer_path /data0/csw/CodeGeeX/codegeex/tokenizer/  --output_prefix pt_data/my_data   --language python  --mode pretrain  --seq_len 128
    # ② python codegeex/data/process_pretrain_dataset.py  --dataset_path sft_data/my_data.jsonl  --tokenizer_path /data0/csw/CodeGeeX/codegeex/tokenizer/  --output_prefix sft_data/my_data                     --mode sft       --seq_len 128
    fire.Fire(main)
