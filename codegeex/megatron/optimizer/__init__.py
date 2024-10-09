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

from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD

from codegeex.megatron import get_args
from codegeex.megatron.model import LayerNorm

from deepspeed.runtime.utils import see_memory_usage

from .grad_scaler import ConstantGradScaler, DynamicGradScaler
from .optimizer import Float16OptimizerWithFloat16Params, FP32Optimizer


def _get_params_for_weight_decay_optimization(modules):
    """Divide params into with-weight-decay and without-weight-decay groups.
    Layernorms and baises will have no weight decay but the rest will.
    """

    # modules: [ CodeGeeXModel(...) ]
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    for module in modules:
        for module_ in module.modules():
            # model.modules() 方法返回的是迭代器 iterator
            # model 的 modules() 方法会将整个模型的所有构成（包括包装层、单独的层、自定义层等）由浅入深依次遍历出来
            if isinstance(module_, LayerNorm):
                no_weight_decay_params["params"].extend(
                    [p for p in list(module_._parameters.values()) if p is not None]
                )
            else:
                weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n != "bias"
                    ]
                )
                no_weight_decay_params["params"].extend(
                    [
                        p
                        for n, p in list(module_._parameters.items())
                        if p is not None and n == "bias"
                    ]
                )

    return weight_decay_params, no_weight_decay_params


def get_megatron_optimizer(model):
    # model: [ CodeGeeXModel(...) ]
    args = get_args()

    # args.cpu_optimizer: False
    if args.cpu_optimizer:
        raise NotImplementedError("need to add cpu adam")
    
    param_groups = _get_params_for_weight_decay_optimization(model)
    # Divide params into with-weight-decay and without-weight-decay groups.
    # Layernorms and baises will have no weight decay but the rest will.
    # weight_decay_params, no_weight_decay_params = param_groups
    # weight_decay_params = {"params": [...]}
    # no_weight_decay_params = {"params": [...], "weight_decay": 0.0}

    see_memory_usage(f"Before Adam Init", force=True)
    # [2024-03-17 14:42:11,700] [INFO] [utils.py:828:see_memory_usage] Before Adam Init
    # [2024-03-17 14:42:11,700] [INFO] [utils.py:829:see_memory_usage] MA 18.04 GB         Max_MA 18.04 GB         CA 18.27 GB         Max_CA 18 GB
    # [2024-03-17 14:42:11,700] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 92.75 GB, percent = 5.0%
    # args.optimizer: "adam"
    if args.optimizer == "adam":
        # from apex.optimizers import FusedAdam as Adam
        # FusedAdam implements Adam algorithm and implements 2 fusions.
        #   * Fusion of the Adam update's elementwise operations
        #   * A multi-tensor apply launch that batches the elementwise updates applied to all the model's parameters into one or a few kernel launches.
        optimizer = Adam(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            # args.weight_decay: 0.1
            betas=(args.adam_beta1, args.adam_beta2),
            # args.adam_beta1: 0.9
            # args.adam_beta2: 0.95
            eps=args.adam_eps,
            # args.adam_eps: 1e-8
        )
    elif args.optimizer == "sgd":
        optimizer = SGD(
            param_groups,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.sgd_momentum,
        )
    else:
        raise Exception("{} optimizer is not supported.".format(args.optimizer))
    see_memory_usage(f"After Adam Init", force=True)
    # [2024-03-17 14:42:11,768] [INFO] [utils.py:828:see_memory_usage] After Adam Init
    # [2024-03-17 14:42:11,769] [INFO] [utils.py:829:see_memory_usage] MA 18.04 GB         Max_MA 18.04 GB         CA 18.27 GB         Max_CA 18 GB
    # [2024-03-17 14:42:11,769] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 87.07 GB, percent = 4.7%

    # Adam 初始化后显存占用没变, 说明 Adam 并未对模型子块的 param 进行复制, 而是使用指向的策略, 不额外占据显存

    # args.deepspeed: True
    if args.deepspeed:
        return optimizer

    # Determine whether the params have main-grad field.
    params_have_main_grad = False
    # args.DDP_impl: "local"
    if args.DDP_impl == "local":
        params_have_main_grad = True

    # args.fp16: True
    if args.fp16 or args.bf16:
        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None
        # Constant loss scale.
        # args.loss_scale: 12.0
        if args.loss_scale:
            grad_scaler = ConstantGradScaler(args.loss_scale)
        # Dynamic loss scale.
        else:
            if args.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=args.initial_loss_scale,
                    min_scale=args.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=args.loss_scale_window,
                    hysteresis=args.hysteresis,
                )

        # Megatron optimizer.
        # Float16OptimizerWithFloat16Params from /home/icksys/csw/CodeGeeX/codegeex/megatron/optimizer/optimizer.py

        see_memory_usage(f"Before Float16OptimizerWithFloat16Params Wrapped", force=True)
        # [2024-03-17 14:49:05,027] [INFO] [utils.py:828:see_memory_usage] Before Float16OptimizerWithFloat16Params Wrapped
        # [2024-03-17 14:49:05,027] [INFO] [utils.py:829:see_memory_usage] MA 18.04 GB         Max_MA 18.04 GB         CA 18.27 GB         Max_CA 18 GB
        # [2024-03-17 14:49:05,028] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 84.24 GB, percent = 4.5%
        return Float16OptimizerWithFloat16Params(
            optimizer,
            # optimizer: FusedAdam(...)
            args.clip_grad,
            # args.clip_grad: 1.0
            args.log_num_zeros_in_grad,
            # args.log_num_zeros_in_grad: False
            params_have_main_grad,
            # params_have_main_grad: True
            args.bf16,
            # args.bf16: False
            grad_scaler,
            # grad_scaler: ConstantGradScaler(12.0)
        )

    # FP32.
    return FP32Optimizer(
        optimizer, args.clip_grad, args.log_num_zeros_in_grad, params_have_main_grad
    )
