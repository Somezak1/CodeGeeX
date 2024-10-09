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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_world_size
from .mappings import copy_to_tensor_model_parallel_region
from .mappings import gather_from_tensor_model_parallel_region
from .mappings import reduce_from_tensor_model_parallel_region
from .mappings import scatter_to_tensor_model_parallel_region
from .random import get_cuda_rng_tracker
from .utils import divide
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from codegeex.megatron import get_args
import deepspeed.runtime.activation_checkpointing.checkpointing as ds_checkpointing


_MODEL_PARALLEL_ATTRIBUTE_DEFAULTS = {
    "tensor_model_parallel": False,
    "partition_dim": -1,
    "partition_stride": 1,
}


def param_is_not_tensor_parallel_duplicate(param):
    return (
        hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel
    ) or (get_tensor_model_parallel_rank() == 0)


def set_tensor_model_parallel_attributes(tensor, is_parallel, dim, stride):
    # Make sure the attributes are not set.
    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        assert not hasattr(tensor, attribute)
    # Set the attributes.
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


def set_defaults_if_not_set_tensor_model_parallel_attributes(tensor):
    def maybe_set(attribute, value):
        if not hasattr(tensor, attribute):
            setattr(tensor, attribute, value)

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_set(attribute, _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS[attribute])


def copy_tensor_model_parallel_attributes(destination_tensor, source_tensor):
    def maybe_copy(attribute):
        if hasattr(source_tensor, attribute):
            setattr(destination_tensor, attribute, getattr(source_tensor, attribute))

    for attribute in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        maybe_copy(attribute)


def _initialize_affine_weight_gpu(weight, init_method, partition_dim, stride=1):
    """
    GPU 版权重初始化。特别关注设置随机种子部分
    1 assert weight 不含以下属性
        "tensor_model_parallel"
        "partition_dim"
        "partition_stride"
    2 之后为 weight 添加以下属性/值
	    "tensor_model_parallel": True,
	    "partition_dim": partition_dim,
	    "partition_stride": stride,
	3 初始化 weight
    """
    """Initialize affine weight for model parallel on GPU."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    if ds_checkpointing.is_configured():
        global get_cuda_rng_tracker
        get_cuda_rng_tracker = ds_checkpointing.get_cuda_rng_tracker

    # 令 TP 组内的进程拥有不同的随机种子
    with get_cuda_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(
    weight,
    output_size,
    input_size,
    per_partition_size,
    partition_dim,
    init_method,
    stride=1,
    return_master_weight=False,
):
    """
    CPU 版权重初始化
    1 assert weight 不含以下属性
        "tensor_model_parallel"
        "partition_dim"
        "partition_stride"
    2 之后为 weight 添加以下属性/值
	    "tensor_model_parallel": True,
	    "partition_dim": partition_dim,
	    "partition_stride": stride,
	3 初始化一个 (output_size, input_size) 形状的 torch.fp16 数组 master_weight
	4 将 master_weight 沿维度 partition_dim 切分, 并将进程对应那份矩阵的值赋给 weight
    """
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    set_tensor_model_parallel_attributes(
        tensor=weight, is_parallel=True, dim=partition_dim, stride=stride
    )

    # Initialize master weight
    master_weight = torch.empty(
        output_size, input_size, dtype=torch.float, requires_grad=False
    )
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(
        master_weight, per_partition_per_stride_size, dim=partition_dim
    )
    rank = get_tensor_model_parallel_rank()
    world_size = get_tensor_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim, init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        # vocab_size
        self.num_embeddings = num_embeddings
        # hidden_state
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.0
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        # 当前进程所在 TP 组进程总数
        self.tensor_model_parallel_size = get_tensor_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        # 根据当前进程在 TP 组中的序号, 确定其所需维护的 WE 部分, 沿着 vocab 维度对 WE 进行切割
        # 例如, 进程 id=0, 维护词表序号 [0,5) 范围内的数据；进程 id=1, 维护 [5,10)
        (
            self.vocab_start_index,
            self.vocab_end_index,
        ) = VocabUtility.vocab_range_from_global_vocab_size(
            self.num_embeddings,
            get_tensor_model_parallel_rank(),
            self.tensor_model_parallel_size,
        )
        # 计算当前进程维护的词表大小
        self.num_embeddings_per_partition = (
            self.vocab_end_index - self.vocab_start_index
        )

        # Allocate weights and initialize.
        # 对 WE 做初始化
        # 读取预训练参数配置
        args = get_args()
        # args.use_cpu_initialization: None
        if args.use_cpu_initialization:
            # CPU 上做初始化
            # 在 CPU 上先生成一个完整的 WE
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    dtype=args.params_dtype,
                    # args.params_dtype: torch.float16,
                )
            )
            # 对 CPU 上的 WE 做切割（随机种子在初始化分布式中已设定好, 不用变）
            _initialize_affine_weight_cpu(
                self.weight,
                self.num_embeddings,
                self.embedding_dim,
                self.num_embeddings_per_partition,
                0,
                init_method,
                # 初始化权重的方法, 例如 xavier 之类
            )
        else:
            # 在 GPU 上做初始化
            # 生成一个切割好的 WE
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype,
                    # args.params_dtype: torch.float16
                )
            )
            # 在 GPU 上做初始化, 注意 TP 组内不同进程采用不同的随机种子, TP 组间对应进程采用相同的随机种子
            _initialize_affine_weight_gpu(
                self.weight, init_method, partition_dim=0, stride=1
            )

    def forward(self, input_):
        """定义输入 X 过 WE 的计算方法, 输出结果已经过 AllReduce"""
        if self.tensor_model_parallel_size > 1:
            # 如果在当前进程维护的 WE 上, 找不到对应的单词, 那么对应位置就赋 0
            # 例如当前的数据的 tokenid 是: [2, 7, 1, 5, 4]
            # 若当前维护的词表是 [0, 1, 2, 3] (start_index=0, end_index = 4),
            # 则mask之后的数据为 [2, 0, 1, 0, 0]
            # 若当前维护的词表是 [4, 5, 6, 7] (start_index=4, end_index=8),
            # 则mask之后的数据为 [0, 7, 0, 5, 4]
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (
                input_ >= self.vocab_end_index
            )
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.

        # 输入 X, 过当前进程维护的部分 WE 的结果
        output_parallel = F.embedding(
            masked_input,
            # masked_input: tensor containing indices into the embedding matrix
            self.weight,
            # self.weight: 切割好的 word embedding 的权重
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        # Mask the output embedding.
        # 当前词表不维护的部分, 都设为 0
        if self.tensor_model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        # 将 TP 组各 GPU 上的结果做 AllReduce
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        input_size,
        # input_size: W 的第一个维度
        output_size,
        # output_size: W 的第二个维度
        bias=True,
        # bias: 是否需要引入 bias
        gather_output=True,
        # gather_output: 决定是否要将 Y1 和 Y2 做 all-gather
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        params_dtype=None,
        skip_init=False,
        device=None,
    ):
        # ParallelSelfAttention 中的 ColumnParallelLinear 初始化时参数为
        # input_size: h
        # output_size: h
        # gather_output: False
        # init_method: init_method_normal(0.02)
        # 其余参数为缺省值

        # ParallelMLP 中的 ColumnParallelLinear 初始化时参数为
        # input_size: h
        # output_size: 4h
        # gather_output: False
        # init_method: init_method_normal(0.02)
        # 其余参数为缺省值

        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        # 当前进程所在 TP 组的总进程数
        world_size = get_tensor_model_parallel_world_size()
        # 每块 GPU 上维护的 hidden_size 的大小, 等于原 hidden_zize / TP 组总进程数
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.device = device
        
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if not skip_init:
            # args.use_cpu_initialization: None
            if args.use_cpu_initialization:
                # CPU 上初始化
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                    )
                )
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                )
                # 初始化后权重就在内存上
            else:
                # GPU 上初始化
                self.weight = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        self.input_size,
                        device=self.device if self.device is not None else torch.cuda.current_device(),
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                        # args.params_dtype: torch.float16
                    )
                )
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=0, stride=stride
                )
                # 初始化后权重就在对应 gpu 上
        else:
            self.register_parameter("weight", None)

        # 对 bias 做处理, 道理同 weight
        # skip_init: False
        if bias and not skip_init:
            if args.use_cpu_initialization:
                # CPU 上初始化
                self.bias = Parameter(
                    torch.empty(self.output_size_per_partition, 
                                dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype)
                )
            else:
                # GPU 上初始化
                self.bias = Parameter(
                    torch.empty(
                        self.output_size_per_partition,
                        device=self.device if self.device is not None else torch.cuda.current_device(),
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                    )
                )
            # bias.shape: [self.output_size_per_partition]
            set_tensor_model_parallel_attributes(self.bias, True, 0, stride)
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        # 定义列切割中的 f 算子
        # 调用 copy_to_tensor_model_parallel_region 则新建一个 _CopyToModelParallelRegion实例（见下）
        # Set up backprop all-reduce.
        input_parallel = copy_to_tensor_model_parallel_region(input_)
        # Matrix multiply.

        # 定义 bias
        # self.skip_bias_add: False
        bias = self.bias if not self.skip_bias_add else None
        # X * 切割好的权重
        output_parallel = F.linear(input_parallel, self.weight, bias)
        # 决定是否要对每个进程上的输出结果做 All-Reduce
        if self.gather_output:
            # All-gather across the partitions.
            # 定义列切割中的 g 算子
            # 调用 gather_from_tensor_model_parallel_region 则新建一个 _GatherFromModelParallelRegion 实例（见下）
            output = gather_from_tensor_model_parallel_region(output_parallel)
            # 把各 GPU 上的输出按照列 gather 起来后, 作为最终输出
        else:
            output = output_parallel
            # 否则最终输出还是自己那块 GPU 算的结果
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where bias
                       can be fused with other elementwise operations. we skip
                       adding bias but instead return it.
    """

    def __init__(
        self,
        input_size,
        output_size,
        bias=True,
        input_is_parallel=False,
        init_method=init.xavier_normal_,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        params_dtype=None,
        skip_init=False,
        device=None,
    ):
        # ParallelSelfAttention 中的 RowParallelLinear 初始化时参数
        # input_size: h
        # output_size: h
        # input_is_parallel: True
        # init_method: scaled_init_method_normal(0.02, 39)
        # skip_bias_add: True
        # 其余参数为缺省值

        # ParallelMLP 中的 RowParallelLinear 初始化时参数
        # input_size: 4h
        # output_size: h
        # input_is_parallel: True
        # init_method: scaled_init_method_normal(0.02, 39)
        # 其余参数为缺省值

        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add
        self.params_dtype = params_dtype
        self.device = device
        
        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        # skip_init: False
        if not skip_init:
            # args.use_cpu_initialization: None
            if args.use_cpu_initialization:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size,
                        self.input_size_per_partition,
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                    )
                )
                # args.params_dtype: torch.float16
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                )
            else:
                self.weight = Parameter(
                    torch.empty(
                        self.output_size,
                        self.input_size_per_partition,
                        device=self.device if self.device is not None else torch.cuda.current_device(),
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                    )
                )
                _initialize_affine_weight_gpu(
                    self.weight, init_method, partition_dim=1, stride=stride
                )
        else:
            self.register_parameter("weight", None)

        # skip_init: False
        if bias and not skip_init:
            if args.use_cpu_initialization:
                self.bias = Parameter(
                    torch.empty(self.output_size, 
                                dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype)
                )
            else:
                self.bias = Parameter(
                    torch.empty(
                        self.output_size,
                        device=self.device if self.device is not None else torch.cuda.current_device(),
                        dtype=self.params_dtype if self.params_dtype is not None else args.params_dtype,
                    )
                )
            # self.bias.shape: [self.output_size]
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter("bias", None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        # ParallelSelfAttention 中的 RowParallelLinear 定义时的 self.skip_bias_add=True
        # 在之后的代码中会进行 dropout(output+output_bias) 的操作, 但由于 output_bias 始终为 0 且不更新参数
        # 因此 dropout(output+output_bias) 等价于 dropout(output)

        # ParallelMLP 中的 RowParallelLinear 定义时的 self.skip_bias_add=False
        # 但同样因为 output_bias 始终为 0 且不更新参数

        # 因此 ParallelSelfAttention 中的 RowParallelLinear 和 ParallelMLP 中的 RowParallelLinear 没啥区别
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
