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

from abc import ABC
from abc import abstractmethod

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from codegeex.megatron import mpu
from .module import MegatronModule


class MemoryBuffer:
    def __init__(self, numel, dtype):
        self.numel = numel
        self.dtype = dtype
        self.data = torch.zeros(
            self.numel,
            dtype=self.dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
        # 在当前GPU上初始化一块空间

    def zero(self):
        """Reset the buffer to zero."""
        # 空间没释放, 只是让值归零
        self.data.zero_()

    def get(self, shape, start_index):
        """Return a tensor with the input `shape` as a view into the
        1-D data starting at `start_index`."""
        end_index = start_index + shape.numel()
        assert end_index <= self.numel, "requested tensor is out of the buffer range."
        buffer_tensor = self.data[start_index:end_index]
        buffer_tensor = buffer_tensor.view(shape)
        return buffer_tensor


class DistributedDataParallelBase(MegatronModule, ABC):
    """Abstract class for DDP."""

    def __init__(self, module):
        super(DistributedDataParallelBase, self).__init__()
        # Keep a pointer to the model.
        # module: Float16Module( CodeGeeXModel(...) )
        self.module = module

    @abstractmethod
    def allreduce_gradients(self):
        pass

    def forward(self, *inputs, **kwargs):
        # 对前向计算没有影响
        return self.module(*inputs, **kwargs)

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


class DistributedDataParallel(DistributedDataParallelBase):
    """DDP with contiguous buffers options to store and accumulate gradients.
    This class:
        - has the potential to reduce memory fragmentation.
        - provides the option to do the gradient accumulation
          in a type other than the params type (for example fp32)

    Arguments:
        module: input model.
        accumulate_allreduce_grads_in_fp32: if true do the gradient accumulation
            and the gradient all-reduce all in in float32. If this option is
            true, we require `use_contiguous_buffers` to be true too.
        use_contiguous_buffers: if true, use a contiguous buffer to store the
            gradients.
    """

    def __init__(
            self, module, accumulate_allreduce_grads_in_fp32, use_contiguous_buffers
    ):
        # module: Float16Module( CodeGeeXModel(...) )
        # accumulate_allreduce_grads_in_fp32: True
        # use_contiguous_buffers: True

        super(DistributedDataParallel, self).__init__(module)

        # 关于 嵌套model 的 for param in 嵌套model.parameters() 其实遍历的是最底层模型的参数
        # class MyNet(nn.Module):
        #     def __init__(self):
        #         super().__init__()
        #         self.conv1 = nn.Conv2d(2, 2, 3)
        #         self.conv2 = nn.Conv2d(2, 2, 3)
        #         self.conv3 = nn.Conv2d(2, 2, 3)
        #
        #     def forward(self, x):
        #         x = self.conv1(x)
        #         x = self.conv2(x)
        #         x = self.conv3(x)
        #
        #         return x
        #
        # class MyNet2(nn.Module):
        #     def __init__(self, module):
        #         super(MyNet2, self).__init__()
        #         self.abc = module
        #
        #     def forward(self, *inputs, **kwargs):
        #         return self.abc(*inputs, **kwargs)
        #
        #
        # model = MyNet()
        # model2 = MyNet2(model)
        #
        # for param in model.parameters():
        #     print(hex(id(param)))
        # print("=====")
        # for param in model2.parameters():
        #     print(hex(id(param)))


        self.accumulate_allreduce_grads_in_fp32 = accumulate_allreduce_grads_in_fp32
        self.use_contiguous_buffers = use_contiguous_buffers
        # If we are using fp32-accumulate-allreduce explicitly
        # this means we need main grads in a continuous buffer.
        if self.accumulate_allreduce_grads_in_fp32:
            assert self.use_contiguous_buffers

        # ===================================
        # Rest of this part applies only to
        # the case we use continuous buffers.
        # ===================================
        self._grad_buffers = None
        # self.use_contiguous_buffers: True
        if self.use_contiguous_buffers:
            self._grad_buffers = {}

            # Simple function to define buffer type.
            def _get_buffer_type(param):
                return (
                    torch.float
                    # torch.float: torch.float32
                    if self.accumulate_allreduce_grads_in_fp32
                    else param.dtype
                )

            # First calculate total number of elements per type.
            # 统计当前进程对应模型子块中参数的数量
            type_num_elements = {}
            for param in self.module.parameters():
                # 关于 嵌套model 的 for param in xxx.parameters() 其实遍历的是最底层模型的参数
                # dtype(param): <class 'torch.nn.parameter.Parameter'>
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    # dtype: torch.float32
                    type_num_elements[dtype] = (
                            type_num_elements.get(dtype, 0) + param.data.nelement()
                            # .nelement(): 统计张量中元素的个数
                    )

            # Allocate the buffer.
            for dtype, num_elements in type_num_elements.items():
                # dtype: torch.float32
                # num_elements: 3227279360, 当前进程对应子模块的参数数量
                # MemoryBuffer 大约占用 12.02G
                self._grad_buffers[dtype] = MemoryBuffer(num_elements, dtype)

            # Assume the back prop order is reverse the params order,
            # store the start index for the gradients.
            for param in self.module.parameters():
                if param.requires_grad:
                    dtype = _get_buffer_type(param)
                    type_num_elements[dtype] -= param.data.nelement()
                    # 在此之前 param.main_grad 并不存在
                    param.main_grad = self._grad_buffers[dtype].get(
                        param.data.shape, type_num_elements[dtype]
                    )
            # 假设module中有3个子模块A/B/C, 分别有20/50/30个参数, 总计100个参数, 即type_num_elements["fp32"]=100
            # 那么程序就会开辟一块可以容纳100个元素的连续空间作为缓存, 每个元素占用4个字节, self._grad_buffers["fp32"]指向该缓存空间
            # 则 A.main_grad 指向 self._grad_buffers["fp32"][80:100] 这段连续空间
            # 则 B.main_grad 指向 self._grad_buffers["fp32"][30:50] 这段连续空间
            # 则 C.main_grad 指向 self._grad_buffers["fp32"][0:30] 这段连续空间

            # Backward hook.
            # Accumalation function for the gradients. We need
            # to store them so they don't go out of scope.
            self.grad_accs = []
            # Loop over all the parameters in the model.
            for param in self.module.parameters():
                # 还没训练时 param.grad 为 None
                if param.requires_grad:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator functtion.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_param_hook(param))
                    self.grad_accs.append(grad_acc)

    def _make_param_hook(self, param):
        """Create the all-reduce hook for backprop."""

        # Hook used for back-prop.
        def param_hook(*unused):
            # Add the gradient to the buffer.
            # 当梯度计算完成后, 执行以下操作
            if param.grad.data is not None:
                # 将计算出的梯度添加到连续缓冲区中
                # param.grad.data.dtype: torch.float16
                param.main_grad.add_(param.grad.data)
                # Now we can deallocate grad memory.
                # 释放原始梯度占用的内存, 因为梯度已经被累积到缓冲区了
                param.grad = None

        return param_hook

    def zero_grad_buffer(self):
        """Set the grad buffer data to zero. Needs to be called at the
        beginning of each iteration."""
        assert self._grad_buffers is not None, "buffers are not initialized."
        # 将缓冲区归零, 但并不清除缓冲区
        for _, buffer_ in self._grad_buffers.items():
            # buffer_: MemoryBuffer(...)
            buffer_.zero()

        # 经测试, 可以凭空给 param 添加新的域 (显存占用会增加), 同时也可以让域指向 None 从而释放显存
        # from deepspeed.runtime.utils import see_memory_usage
        # see_memory_usage(f"Before Init Model", force=True)
        #
        # model = MyNet().cuda()
        #
        # see_memory_usage(f"After Init Model / Before add main_grad", force=True)
        #
        # for param in model.parameters():
        #     param.main_grad = torch.zeros(100000000, dtype=torch.float32, device=torch.cuda.current_device(), requires_grad=False)
        #
        # see_memory_usage(f"After add main_grad / Before delete main_grad", force=True)
        #
        # for param in model.parameters():
        #     param.main_grad = None
        #
        # see_memory_usage(f"After delete main_grad", force=True)

    def allreduce_gradients(self):
        """Reduce gradients across data parallel ranks."""
        # If we have buffers, simply reduce the data in the buffer.
        if self._grad_buffers is not None:
            for _, buffer_ in self._grad_buffers.items():
                # 将数据并行组各模型的累积梯度求平均
                # 数据并行组各模型原始的权重是一样的
                buffer_.data /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    buffer_.data, group=mpu.get_data_parallel_group()
                )
        else:
            # Otherwise, bucketize and all-reduce
            buckets = {}
            # Pack the buckets.
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = param.data.type()
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
                    param.main_grad = param.grad

            # For each bucket, all-reduce and copy all-reduced grads.
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                coalesced /= mpu.get_data_parallel_world_size()
                torch.distributed.all_reduce(
                    coalesced, group=mpu.get_data_parallel_group()
                )
                for buf, synced in zip(
                        grads, _unflatten_dense_tensors(coalesced, grads)
                ):
                    buf.copy_(synced)
