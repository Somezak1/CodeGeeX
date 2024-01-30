from typing import *
from time import perf_counter

from codegeex.data.data_utils import sliding_window
from codegeex.data.types import PromptSample, LabelSample


class PromptDatasetProcessor(object):
    def __init__(
        self,
        tokenize: Callable,
        pad_token: int,
        keep_order: bool = False,
        max_seq_len: int = 2048,
        sliding_stride: int = 200,
        discard_overlong: bool = True,
        eod_token: int = None, 
        preprocess: Callable = None,
    ):
        # tokenize: tokenizer.encode_code
        # pad_token: tokenizer.eos_token_id
        # max_seq_len: 128 + 1
        # discard_overlong: False
        # sliding_stride: 200
        # eod_token: tokenizer.eos_token_id

        super(PromptDatasetProcessor, self).__init__()
        self._keep_order = keep_order
        self._max_seq_len = max_seq_len
        self._sliding_stride = sliding_stride
        self._tokenize = tokenize
        self._pad_token = pad_token
        self._discard_overlong = discard_overlong
        self._eod_token = eod_token
        self._preprocess = preprocess

        self.doc_processed = 0
        self.doc_generated = 0
        self.start_time = 0

    def pad_seq(self, prompt_tokens: List[int], code_tokens: List[int], extra: dict = None) -> Dict[str, List[int]]:
        total_length = len(prompt_tokens) + len(code_tokens)
        assert total_length <= self._max_seq_len, f"padding sequence: {total_length} > {self._max_seq_len}"
        pad_len = self._max_seq_len - total_length
        input_ids = prompt_tokens + code_tokens + [self._pad_token] * pad_len
        attention_mask = [1] * len(prompt_tokens) + [1] * len(code_tokens) + [0] * pad_len
        labels = [-100] * len(prompt_tokens) + code_tokens + [-100] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def process_sample(self, sample: PromptSample) -> Iterable[Dict[str, List[int]]]:
        """
        Process a sample.
        """
        prompt_tokens = self._tokenize(sample.prompt)
        # ①
        # sample.prompt: ''
        # sample.prompt after encode_whitespaces: ''
        # prompt_tokens: []
        # ②
        # sample.prompt: 'Please translate below code into python:\nint main() {\n    int a = 0;\n    for (int i = 0; i < a; i++) {\n        std::cout << i << std::endl;\n    }\n    return 0;\n}\npython:\n'
        # sample.prompt after encode_whitespaces: 'Please translate below code into python:\nint main() {\n<|extratoken_12|>int a = 0;\n<|extratoken_12|>for (int i = 0; i < a; i++) {\n<|extratoken_16|>std::cout << i << std::endl;\n<|extratoken_12|>}\n<|extratoken_12|>return 0;\n}\npython:\n'
        # prompt_tokens: [5492, 15772, 2174, 2438, 656, 21015, 25, 198, 600, 1388, 3419, 1391, 198, 50268, 600, 257, 796, 657, 26, 198, 50268, 1640, 357, 600, 1312, 796, 657, 26, 1312, 1279, 257, 26, 1312, 29577, 1391, 198, 50272, 19282, 3712, 66, 448, 9959, 1312, 9959, 14367, 3712, 437, 75, 26, 198, 50268, 92, 198, 50268, 7783, 657, 26, 198, 92, 198, 29412, 25, 198]
        code_tokens = self._tokenize(sample.code)
        # ①
        # sample.code: '# language: Python\ndef main():\n    a = 0\n    for i in range(a):\n        print(i)\n    return'
        # sample.code after encode_whitespaces: '# language: Python\ndef main():\n<|extratoken_12|>a = 0\n<|extratoken_12|>for i in range(a):\n<|extratoken_16|>print(i)\n<|extratoken_12|>return'
        # code_tokens: [2, 3303, 25, 11361, 198, 4299, 1388, 33529, 198, 50268, 64, 796, 657, 198, 50268, 1640, 1312, 287, 2837, 7, 64, 2599, 198, 50272, 4798, 7, 72, 8, 198, 50268, 7783]
        # ②
        # sample.code: 'def main():\n    a = 0\n    for i in range(a):\n        print(i)\n    return'
        # sample.code after encode_whitespaces: 'def main():\n<|extratoken_12|>a = 0\n<|extratoken_12|>for i in range(a):\n<|extratoken_16|>print(i)\n<|extratoken_12|>return'
        # code_tokens: [4299, 1388, 33529, 198, 50268, 64, 796, 657, 198, 50268, 1640, 1312, 287, 2837, 7, 64, 2599, 198, 50272, 4798, 7, 72, 8, 198, 50268, 7783]

        if self._eod_token is not None:
            code_tokens.append(self._eod_token)

        if len(prompt_tokens) + len(code_tokens) > self._max_seq_len:
            if self._discard_overlong:
                return
            for p, t in sliding_window(prompt_tokens, code_tokens, self._max_seq_len, self._sliding_stride, self._sliding_stride):
                yield self.pad_seq(p, t)
        else:
            yield self.pad_seq(prompt_tokens, code_tokens, extra=sample.extra)

    def process_sample_strict(self, sample: PromptSample) -> List[Dict[str, List[int]]]:
        """
        Instead of processing lazily, we turn the iterable into a list.
        """
        # sample: [{"input_ids": ..., "attention_mask": ..., "labels": ...}]
        if sample is None:
            return None
        
        return list(self.process_sample(sample))

    def process_sample_(self, sample) -> List[Dict[str, List[int]]]:
        prompt_sample = self._preprocess(sample)
        return self.process_sample_strict(prompt_sample)

    def report(self):
        duration = perf_counter() - self.start_time
        process_speed = self.doc_processed * 1.0 / duration
        gen_speed = self.doc_generated * 1.0 / duration
        print(f">>> processed: {self.doc_processed} in {duration:.2f}s, speed: {process_speed:.2f} docs/s")
        print(f"... generated: {self.doc_generated} in {duration:.2f}s, speed: {gen_speed:.2f} docs/s")



class LabelDatasetProcessor(object):
    def __init__(
        self,
        tokenize: Callable,
        pad_token: int,
        keep_order: bool = False,
        max_seq_len: int = 2048,
        sliding_stride: int = 200,
        discard_overlong: bool = True,
        eod_token: int = None, 
        preprocess: Callable = None,
    ):
        super(LabelDatasetProcessor, self).__init__()
        self._keep_order = keep_order
        self._max_seq_len = max_seq_len
        self._sliding_stride = sliding_stride
        self._tokenize = tokenize
        self._pad_token = pad_token
        self._discard_overlong = discard_overlong
        self._eod_token = eod_token
        self._preprocess = preprocess

        self.doc_processed = 0
        self.doc_generated = 0
        self.start_time = 0

    def pad_seq(self, prompt_tokens: List[int], label: int, extra: dict = None) -> Dict[str, List[int]]:
        total_length = len(prompt_tokens) 
        assert total_length <= self._max_seq_len, f"padding sequence: {total_length} > {self._max_seq_len}"
        pad_len = self._max_seq_len - total_length
        input_ids = prompt_tokens +  [self._pad_token] * pad_len
        attention_mask = [1] * len(prompt_tokens) + [0] * pad_len
        label = [label]

        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "length": [len(prompt_tokens)],
                "labels": label
        }
    def process_sample(self, sample: LabelSample) -> Iterable[Dict[str, List[int]]]:
        """
        Process a sample.
        """
        prompt_tokens = self._tokenize(sample.prompt)
        label = sample.label

        
        if len(prompt_tokens) > self._max_seq_len:
            if self._discard_overlong:
                return
            prompt_tokens=prompt_tokens[-self._max_seq_len:]
        
        yield self.pad_seq(prompt_tokens, label, extra=sample.extra)

    def process_sample_strict(self, sample: LabelSample) -> List[Dict[str, List[int]]]:
        """
        Instead of processing lazily, we turn the iterable into a list.
        """
        if sample is None:
            return None
        
        return list(self.process_sample(sample))

    def process_sample_(self, sample) -> List[Dict[str, List[int]]]:
        prompt_sample = self._preprocess(sample)
        return self.process_sample_strict(prompt_sample)

    def report(self):
        duration = perf_counter() - self.start_time
        process_speed = self.doc_processed * 1.0 / duration
        gen_speed = self.doc_generated * 1.0 / duration
        print(f">>> processed: {self.doc_processed} in {duration:.2f}s, speed: {process_speed:.2f} docs/s")
        print(f"... generated: {self.doc_generated} in {duration:.2f}s, speed: {gen_speed:.2f} docs/s")
