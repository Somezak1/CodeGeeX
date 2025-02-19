# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron Module"""

import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter

from codegeex.megatron import get_args
from codegeex.megatron import mpu


_FLOAT_TYPES = (torch.FloatTensor, torch.cuda.FloatTensor)
_HALF_TYPES = (torch.HalfTensor, torch.cuda.HalfTensor)
_BF16_TYPES = (torch.BFloat16Tensor, torch.cuda.BFloat16Tensor)


def param_is_not_shared(param):
    return not hasattr(param, "shared") or not param.shared


class MegatronModule(torch.nn.Module):
    """Megatron specific extensions of torch Module with support
    for pipelining."""

    def __init__(self, share_word_embeddings=True):
        super(MegatronModule, self).__init__()
        # input 和 output 是否要共享一套 WE
        self.share_word_embeddings = share_word_embeddings

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        """Use this function to override the state dict for
        saving checkpoints."""
        # 模型训练中, 及时将参数保存到指定位置（设置 checkpoint）
        # 这样在训练出问题时, 可以从 checkpoint 点重新 load 参数, 继续训练
        return self.state_dict(destination, prefix, keep_vars)

    def word_embeddings_weight(self):
        if mpu.is_pipeline_first_stage(ignore_virtual=True):
            return self.language_model.embedding.word_embeddings.weight
        if mpu.is_pipeline_last_stage(ignore_virtual=True):
            if not self.share_word_embeddings:
                # 强制要求共享一套 embedding
                raise Exception(
                    "word_embeddings_weight() called for last "
                    "stage, but share_word_embeddings is false"
                )
            # 参见 initialize_word_embeddings 中 WE 的定义
            return self.word_embeddings.weight
        # 如果当前进程是 PP 组的中间进程, 则其上未维护 WE, 因此当然获取不到
        raise Exception(
            "word_embeddings_weight() should be " "called for first and last stage only"
        )

    def initialize_word_embeddings(self, init_method_normal):
        args = get_args()
        # 强制share embeddingg
        if not self.share_word_embeddings:
            raise Exception(
                "initialize_word_embeddings() was called but "
                "share_word_embeddings is false"
            )

        # This function just initializes the word embeddings in the final stage
        # when we are using pipeline parallelism. If we aren't using pipeline
        # parallelism there is nothing to do.
        # PP 组并行度为 1 时, 第一层和最后一层都在一块 GPU 上, 天然共享 WE, 无需做强制
        if args.pipeline_model_parallel_size == 1:
            return

        # Parameters are shared between the word embeddings layer, and the
        # heads at the end of the model. In a pipelined setup with more than
        # one stage, the initial embedding layer and the head are on different
        # workers, so we do the following:
        # 1. Create a second copy of word_embeddings on the last stage, with
        #    initial parameters of 0.0.
        # 2. Do an all-reduce between the first and last stage to ensure that
        #    the two copies of word_embeddings start off with the same
        #    parameter values.
        # 3. In the training loop, before an all-reduce between the grads of
        #    the two word_embeddings layers to ensure that every applied weight
        #    update is the same on both stages.
        # ---------------------------------------------------
        # 如果流水线并行的度不为 1 时, 依次做三件事:
        # 【初始化时】:
        # 1、在 PP 组最后一个进程上初始化一个 WE, 令其取值全为 0
        # 2、在 PP 组第一个进程与最后一个进程间做一次 AllReduce, 保证两者的 WE 完全一致
        # 【训练时】:
        # 3、每次想在 PP 组第一个/最后一个进程上使用 WE 时, 要做一次通信, 保证两者用的 WE 完全一致
        if mpu.is_pipeline_last_stage():
            # 若当前进程是 PP 组最后一个进程
            assert not mpu.is_pipeline_first_stage()
            self._word_embeddings_for_head_key = "word_embeddings_for_head"
            # 初始化一个 WE
            # VocabParallelEmbedding 将在下文详细讲解
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.word_embeddings = mpu.VocabParallelEmbedding(
                args.padded_vocab_size,
                # vocab_size
                args.hidden_size,
                # embed_dim
                init_method=init_method_normal(args.init_method_std),
                # 初始化方法（在 model/utils.py 下）
            )
            # 用 0 填充 WE（等待下面做 AllReduce 后取得第一个进程上的 WE）
            self.word_embeddings.weight.data.fill_(0)
            self.word_embeddings.weight.shared = True

        # Ensure that first and last stages have the same initial parameter
        # values.
        if torch.distributed.is_initialized():
            # 若当前进程是 PP 组第一个或最后一个进程
            if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage():
                # 在两进程间做 AllReduce, 保证它们使用的 WE 完全一致
                # mpu.get_embedding_group: 在源码解读 1 中讲过, 是除 DP/TP/PP 之外设置的又一进程组
                # 主要就是用来做关于WE的通讯
                torch.distributed.all_reduce(
                    self.word_embeddings_weight().data, group=mpu.get_embedding_group()
                )
        else:
            print(
                "WARNING! Distributed processes aren't initialized, so "
                "word embeddings in the last layer are not initialized. "
                "If you are just manipulating a model this is fine, but "
                "this needs to be handled manually. If you are training "
                "something is definitely wrong."
            )


def conversion_helper(val, conversion):
    """Apply conversion to val. Recursively apply conversion if `val`
    #is a nested tuple/list structure."""

    # tokens, position_ids, attention_mask = val
    # isinstance(val, (tuple, list)): True
    if not isinstance(val, (tuple, list)):
        return conversion(val)
    rtn = [conversion_helper(v, conversion) for v in val]
    if isinstance(val, tuple):
        rtn = tuple(rtn)
    # rtn: [conversion_helper(tokens, conversion), conversion_helper(position_ids, conversion), conversion_helper(attention_mask, conversion)]
    # 即 [conversion(tokens), conversion(position_ids), conversion(attention_mask)]
    return rtn


def fp32_to_float16(val, float16_convertor):
    """Convert fp32 `val` to fp16/bf16"""

    # tokens, position_ids, attention_mask = val
    # tokens:           torch.Size([2, 128])            torch.int64
    # position_ids:     torch.Size([2, 128])            torch.int64
    # attention_mask:   torch.Size([1, 1, 128, 128])    torch.bool

    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, _FLOAT_TYPES):
            val = float16_convertor(val)
        return val

    # conversion_helper(val, half_conversion): [half_conversion(tokens), half_conversion(position_ids), half_conversion(attention_mask)]
    return conversion_helper(val, half_conversion)


def float16_to_fp32(val):
    """Convert fp16/bf16 `val` to fp32"""

    def float_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if isinstance(val_typecheck, (_BF16_TYPES, _HALF_TYPES)):
            val = val.float()
        return val

    return conversion_helper(val, float_conversion)


class Float16Module(MegatronModule):
    def __init__(self, module, args):
        super(Float16Module, self).__init__()
        # module: CodeGeeXModel(...)

        # args.fp16: True
        if args.fp16:
            self.add_module("module", module.half())
            # self.add_module("module", module.half()) 等价于 self.module = module.half()
            # module.half(): 会将所有参数变为 fp16 类型, 具体来说, LayerNorm 会从 fp32 变为 fp16, LayerNorm 以外的参数原本就是 fp16, 所以没变化

            def float16_convertor(val):
                return val.half()

        elif args.bf16:
            self.add_module("module", module.bfloat16())

            def float16_convertor(val):
                return val.bfloat16()

        else:
            raise Exception("should not be here")

        self.float16_convertor = float16_convertor

    def forward(self, *inputs, **kwargs):
        # tokens, position_ids, attention_mask = inputs
        # labels = kwargs["labels"]
        # tokens:           torch.Size([2, 128])            torch.int64
        # position_ids:     torch.Size([2, 128])            torch.int64
        # attention_mask:   torch.Size([1, 1, 128, 128])    torch.bool
        # labels:           torch.Size([2, 128])            torch.int64

        # mpu.is_pipeline_first_stage(): True
        if mpu.is_pipeline_first_stage():
            inputs = fp32_to_float16(inputs, self.float16_convertor)
            # 因为 inputs 中的元素既不是模型参数也不是 float 类型, 所以 fp32_to_float16(...) 这步对 inputs 没有任何改变
            # tokens, position_ids, attention_mask = inputs
            # tokens:           torch.Size([2, 128])            torch.int64
            # position_ids:     torch.Size([2, 128])            torch.int64
            # attention_mask:   torch.Size([1, 1, 128, 128])    torch.bool
        outputs = self.module(*inputs, **kwargs)
        # outputs: torch.Size([2, 128])  torch.float32

        # mpu.is_pipeline_last_stage(): True
        if mpu.is_pipeline_last_stage():
            outputs = float16_to_fp32(outputs)
            # 因为 outputs 既不是模型参数也不是半精度(本身就是单精度), 所以 float16_to_fp32(...) 这步对 outputs 也没有任何改变
            # outputs: torch.Size([2, 128])  torch.float32
        return outputs

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        return self.module.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)
