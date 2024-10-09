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

import torch

from .initialize import get_tensor_model_parallel_group
from .initialize import get_tensor_model_parallel_rank
from .initialize import get_tensor_model_parallel_src_rank


_MAX_DATA_DIM = 5


def _check_data_types(keys, data, target_dtype):
    """Check that all the keys have the same target data type."""
    for key in keys:
        assert (
            data[key].dtype == target_dtype
        ), "{} has data type {} which " "is different than {}".format(
            key, data[key].dtype, target_dtype
        )


def _build_key_size_numel_dictionaries(keys, data):
    """Build the size on rank 0 and broadcast."""
    max_dim = _MAX_DATA_DIM
    sizes = [0 for _ in range(max_dim) for _ in keys]
    # sizes: [0, 0, 0, 0, 0]
    # keys: ["input_ids"]
    # keys 中有几个元素, sizes 中就有几组 五个零, 比如 keys = ["input_ids", "labels"], 那么 sizes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Pack the sizes on rank zero.
    if get_tensor_model_parallel_rank() == 0:
        offset = 0
        for key in keys:
            # data[key].dim(): 2
            assert data[key].dim() < max_dim, "you should increase MAX_DATA_DIM"
            size = data[key].size()
            # size: [micro_batch_size, actual_seq_length], 此处为 [2, 129]
            for i, s in enumerate(size):
                sizes[i + offset] = s
            # sizes: [2, 128, 0, 0, 0]
            offset += max_dim

    # Move to GPU and broadcast.
    sizes_cuda = torch.cuda.LongTensor(sizes)
    torch.distributed.broadcast(
        sizes_cuda,
        get_tensor_model_parallel_src_rank(),
        group=get_tensor_model_parallel_group(),
    )

    # Move back to cpu and unpack.
    sizes_cpu = sizes_cuda.cpu()
    # sizes_cpu: [2, 129, 0, 0, 0]
    key_size = {}
    key_numel = {}
    total_numel = 0
    offset = 0
    for key in keys:
        i = 0
        size = []
        numel = 1
        while sizes_cpu[offset + i] > 0:
            this_size = sizes_cpu[offset + i]
            size.append(this_size)
            numel *= this_size
            i += 1
        key_size[key] = size
        key_numel[key] = numel
        total_numel += numel
        offset += max_dim

    # 此处 keys = ["input_ids"]
    # key_size: {'input_ids': [2, 129]}
    # key_numel: {'input_ids': 258}
    # total_numel: 258

    # 假如 keys = ["input_ids", "labels"], 且 data["labels"] 的形状也是 [2, 129]
    # key_size: {'input_ids': [2, 129], 'labels': [2, 129]}
    # key_numel: {'input_ids': 258, 'labels': 258}
    # total_numel: 516
    return key_size, key_numel, total_numel


def broadcast_data(keys, data, datatype):
    """Broadcast data from rank zero of each model parallel group to the
    members of the same model parallel group.

    Arguments:
        keys: list of keys in the data disctionary to be broadcasted
        data: data dictionary of string keys and cpu tensor values.
        datatype: torch data type of all tensors in data associated
                  with keys.
    """
    # Build (key, size) and (key, number of elements) dictionaries along
    # with the total number of elements on all ranks.

    # keys: ["input_ids"]
    # _build_key_size_numel_dictionaries(...): 张量并行组 local rank = 0 的进程与张量并行组内其他进程通信, 以散步要传递数据的元信息
    key_size, key_numel, total_numel = _build_key_size_numel_dictionaries(keys, data)
    # 此处 keys = ["input_ids"]
    # key_size: {'input_ids': [2, 129]}
    # key_numel: {'input_ids': 258}
    # total_numel: 258

    # 假如 keys = ["input_ids", "labels"], 且 data["labels"] 的形状也是 [2, 129]
    # key_size: {'input_ids': [2, 129], 'labels': [2, 129]}
    # key_numel: {'input_ids': 258, 'labels': 258}
    # total_numel: 516

    # Pack on rank zero.
    if get_tensor_model_parallel_rank() == 0:
        # Check that all keys have the same data type.
        _check_data_types(keys, data, datatype)
        # Flatten the data associated with the keys
        flatten_data = torch.cat(
            [data[key].contiguous().view(-1) for key in keys], dim=0
        ).cuda()
    else:
        flatten_data = torch.empty(
            total_numel, device=torch.cuda.current_device(), dtype=datatype
        )

    # Broadcast
    # 将展平后的数据从张量并行组 local rank = 0 的进程发送至组内其余进程
    torch.distributed.broadcast(
        flatten_data,
        get_tensor_model_parallel_src_rank(),
        group=get_tensor_model_parallel_group(),
    )

    # Unpack
    output = {}
    offset = 0
    for key in keys:
        size = key_size[key]
        numel = key_numel[key]
        output[key] = flatten_data.narrow(0, offset, numel).view(size)
        offset += numel

    return output
