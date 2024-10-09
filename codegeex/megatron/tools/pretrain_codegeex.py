import os
import subprocess
import torch
import logging

logging.getLogger("torch").setLevel(logging.WARNING)

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from functools import partial

from codegeex.megatron import get_args, print_rank_0, get_timers,get_tokenizer, mpu
from codegeex.megatron.data.prompt_dataset import build_train_valid_test_datasets
from codegeex.megatron.model import CodeGeeXModel  #, CodeGeeXModelPipe
from codegeex.megatron.training import pretrain
from codegeex.megatron.utils import get_ltor_masks_and_position_ids
from codegeex.megatron.utils import average_losses_across_data_parallel_group


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    # pre_process: True
    # post_process: True

    print_rank_0("building GPT model ...")
    see_memory_usage(f"Before Building Model", force=True)
    # [2024-03-12 21:51:03,723] [INFO] [utils.py:828:see_memory_usage] Before Building Model
    # [2024-03-12 21:51:03,724] [INFO] [utils.py:829:see_memory_usage] MA 0.0 GB         Max_MA 0.0 GB         CA 0.0 GB         Max_CA 0 GB
    # [2024-03-12 21:51:03,725] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 60.68 GB, percent = 3.2%

    # torch.cuda.memory_allocated(device=None): Return the current GPU memory occupied by tensors in bytes for a given device.
    # torch.cuda.memory_reserved(device=None): Return the current GPU memory managed by the caching allocator in bytes for a given device.

    args = get_args()
    with deepspeed.zero.Init(
        data_parallel_group=mpu.get_data_parallel_group(),
        remote_device=None if args.remote_device == "none" else args.remote_device,
        # args.remote_device: "none"
        config_dict_or_path=args.deepspeed_config,
        # args.deepspeed_config: "/home/icksys/csw/CodeGeeX/scripts/ds_config.json"
        enabled=args.zero_stage == 3,
        # args.zero_stage: 2
        mpu=mpu,
    ):
        # args.deepspeed: True
        # args.no_pipeline_parallel: True
        if args.deepspeed and not args.no_pipeline_parallel:
            model = CodeGeeXModelPipe(num_tokentypes=0, parallel_output=True)
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(
                torch.ones(
                    (1, args.seq_length, args.seq_length),
                    device=torch.cuda.current_device(),
                )
            ).view(1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = attention_mask < 0.5
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

        else:
            # 根据所属进程子组的不同, 每个进程定义的 model 都是完整 CodeGeeX 模型的一个子块
            model = CodeGeeXModel(
                num_tokentypes=0,
                parallel_output=True,
            )

            # args.load_state: "/home/icksys/csw/CodeGeeX/scripts/mp4_parallel_weights/"
            # 加载预训练权重
            if args.load_state is not None:
                timers = get_timers()
                print_rank_0("Loading warmstarting model states ...")
                timers("load-model-states").start()
                mp_rank = mpu.get_tensor_model_parallel_rank()
                if os.path.isdir(args.load_state):
                    model_path = os.path.join(
                        args.load_state, "mp_rank_{:02d}_model_states.pt".format(mp_rank)
                    )
                else:
                    model_path = args.load_state
                print_rank_0(f"Loading model from {model_path} ...")
                state_dict = torch.load(model_path, map_location="cpu")
                if "module" in state_dict:
                    state_dict = state_dict["module"]  # strip other client states
                # 为当前进程的模型子块加载预训练的模型权重
                model.load_state_dict(state_dict)
                timers("load-model-states").stop()
                timers.log(["load-model-states"])
    see_memory_usage(f"After Building Model", force=True)
    # [2024-03-12 21:51:08,370] [INFO] [utils.py:828:see_memory_usage] After Building Model
    # [2024-03-12 21:51:08,370] [INFO] [utils.py:829:see_memory_usage] MA 5.99 GB         Max_MA 5.99 GB         CA 6.23 GB         Max_CA 6 GB
    # [2024-03-12 21:51:08,371] [INFO] [utils.py:837:see_memory_usage] CPU Virtual Memory:  used = 98.28 GB, percent = 5.3%
    
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["input_ids"]
    datatype = torch.int64

    # Broadcast data.
    # data_iterator 是 torch.utils.data.DataLoader 对象, 遍历时每次返回一个如下字典
    # {
    #     "input_ids": 形状为 [b, s+1] 的一个张量,
    #     "attention_mask": 形状为 [b, s+1] 的一个张量,
    #     "labels": 形状为 [b, s+1] 的一个张量,
    # }

    # 随便拿一个 batch 的 data 看看
    # {'input_ids': tensor([
    #         [    2,  3303,    25, 11361,   198,  4299,  1388, 33529,   198, 50268,
    #             64,   796,  2534,   198, 50268,  1640,  1312,   287,  2837,     7,
    #             64,  2599,   198, 50272,  4798,     7,    72,     8,   198, 50268,
    #           7783, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256],
    #         [    2,  3303,    25, 11361,   198,  4299,  1388, 33529,   198, 50268,
    #             64,   796, 27191,   198, 50268,  1640,  1312,   287,  2837,     7,
    #             64,  2599,   198, 50272,  4798,     7,    72,     8,   198, 50268,
    #           7783, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256,
    #          50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256, 50256]]),
    # 'attention_mask': tensor([
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    #          1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #          0, 0, 0, 0, 0, 0, 0, 0, 0]]),
    # 'labels': tensor([
    #         [    2,  3303,    25, 11361,   198,  4299,  1388, 33529,   198, 50268,
    #             64,   796,  2534,   198, 50268,  1640,  1312,   287,  2837,     7,
    #             64,  2599,   198, 50272,  4798,     7,    72,     8,   198, 50268,
    #           7783, 50256,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100],
    #         [    2,  3303,    25, 11361,   198,  4299,  1388, 33529,   198, 50268,
    #             64,   796, 27191,   198, 50268,  1640,  1312,   287,  2837,     7,
    #             64,  2599,   198, 50272,  4798,     7,    72,     8,   198, 50268,
    #           7783, 50256,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
    #           -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100]])}

    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None

    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["input_ids"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    # tokens.shape: [b, s]
    # labels.shape: [b, s]

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        # tokenizer.eod: 50256
        args.reset_position_ids,
        # args.reset_position_ids: False
        args.reset_attention_mask,
        # args.reset_attention_mask: False
        args.eod_mask_loss,
        # args.eod_mask_loss: False
    )

    # 在此运行脚本下, s=128
    # attention_mask.shape: [1, 1, s, s]
    # attention_mask[0, 0, :10, :10]:
    # tensor([
    #     [False,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    #     [False, False,  True,  True,  True,  True,  True,  True,  True,  True],
    #     [False, False, False,  True,  True,  True,  True,  True,  True,  True],
    #     [False, False, False, False,  True,  True,  True,  True,  True,  True],
    #     [False, False, False, False, False,  True,  True,  True,  True,  True],
    #     [False, False, False, False, False, False,  True,  True,  True,  True],
    #     [False, False, False, False, False, False, False,  True,  True,  True],
    #     [False, False, False, False, False, False, False, False,  True,  True],
    #     [False, False, False, False, False, False, False, False, False,  True],
    #     [False, False, False, False, False, False, False, False, False, False]
    # ], device='cuda:0')
    # 经过验证, attention_mask[0, 0] 就是一个 [s, s] 尺寸的上三角矩阵

    # loss_mask.shape: [b, s]
    # 全 1 矩阵

    # position_ids.shape: [b, s]
    # position_ids[i]: tensor([   0,    1,    2,  ..., s-1], device='cuda:0')

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ["input_ids"]
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b["input_ids"].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
    )

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, output_tensor):
    # loss_mask.shape: [b, s], dtype: torch.float32
    # output_tensor.shape: [b, s], dtype: torch.float32

    losses = output_tensor.float()
    # output_tensor 是一个已经在同一张量并行组内各进程间经过 All Reduce, 未坍缩的损失矩阵, 同一张量并行组内所有进程的 output_tensor 一样
    # output_tensor 代表该进程所属模型并行组的模型在当前 micro_batch_size 数据上的损失矩阵
    loss_mask = loss_mask.view(-1).float()
    # .float() -->  torch.float32
    # .half()  -->  torch.float16
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    # loss: 形如 tensor(3.9353), 代表该进程所属模型并行组 (因此没有流水线并行, 所以张量并行组等价于模型并行组) 的模型在当前 micro_batch_size 数据上的损失值, 此时同一张量并行组内各进程的 loss 一致

    # Reduce loss for logging.
    # averaged_loss 表示当前所有数据并行模型在该 micro_batch_size 数据上的平均损失
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {"lm loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers("batch-generator").start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    timers("batch-generator").stop()
    # 下面是从训练时每次迭代的日志中截取的一段关于迭代用时的输出字段 (训练脚本未使用 deepspeed)
    # time (ms) | forward-compute: 152.33 | backward-compute: 157.91 | backward-params-all-reduce: 101.10 | backward-embedding-all-reduce: 0.03 | optimizer-copy-to-main-grad: 0.72 | optimizer-unscale-and-check-inf: 18.83 | optimizer-clip-main-grad: 28.35 | optimizer-copy-main-to-model-params: 19.37 | optimizer: 129.43 | batch-generator: 0.64
    # forward-compute 和 backward-compute 分别是在一个 global_batch_size 数据上前向传播 和 反向传播 (不包含梯度更新) 的总耗时
    # optimizer: 129.43 表示完成 optimizer.step() 的总耗时, 由 backward-params-all-reduce 到 optimizer-copy-main-to-model-params 以及 参数更新 阶段构成
    # 注意, 从日志中截取的这段输出字段并不包含 optimizer 进行参数更新的耗时
    # batch-generator: 0.64 是从 dataloader 中取下个批次 global_batch_size 数据 (包含取数据, 分发数据) 的耗时

    # b 指的是 micro_batch_size, s 指的是实际输入的序列长度
    # tokens.shape: [b, s], dtype: torch.int64
    # labels.shape: [b, s], dtype: torch.int64
    # loss_mask.shape: [b, s], dtype: torch.float32
    # attention_mask.shape: [1, 1, s, s], dtype: torch.bool
    # position_ids.shape: [b, s], dtype: torch.int64

    # 如果训练脚本不使用 deepspeed, 那么 model: LocalDDP ( Float16Module( CodeGeeXModel(...) ) )
    # 但因为当前训练脚本使用了 deepspeed, 所以 model: DeepSpeedEngine( CodeGeeXModel(...) )
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    # output_tensor.shape: [b, s], dtype: torch.float32, 代表该进程所属模型并行组 (因此没有流水线并行, 所以张量并行组等价于模型并行组) 的模型在当前 micro_batch_size 数据上的损失矩阵
    # output_tensor 是一个已经在同一张量并行组内各进程间经过 All Reduce, 未坍缩的损失矩阵, 同一张量并行组内所有进程的 output_tensor 一样

    return output_tensor, partial(loss_func, loss_mask)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0("> building train, validation, and test datasets " "for GPT ...")
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        # args.data_path: ['/home/icksys/csw/CodeGeeX/pt_data/my_data']
        data_impl=args.data_impl,
        # args.data_impl: 'mmap'
        splits_string=args.split,
        # args.split: [100, 0, 0]
        train_valid_test_num_samples=train_val_test_num_samples,
        # train_val_test_num_samples: [100, 120, 40]
        seq_length=args.seq_length,
        # args.seq_length: 512
        seed=args.seed,
        # args.seed: 1234
        skip_warmup=(not args.mmap_warmup),
        # args.mmap_warmup: False
    )
    # 返回的 train_dataset, valid_dataset, test_dataset 一般都是 PromptDataset 类
    # PromptDataset 类继承自 torch.utils.data.Dataset, 遍历时每次返回一个如下形式的字典, 表示一个样本
    # {
    #     "input_ids": np.array(input_ids, dtype=np.int64),
    #     "attention_mask": np.array(attention_mask, dtype=np.int64),
    #     "labels": np.array(labels, dtype=np.int64),
    # }
    # 当然, 如果 splits_string = [100, 0, 0], 则没有样本用于构建 valid 和 test 数据集, 因此 valid_dataset 和 test_dataset 都为 None
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def command_exists(cmd):
    result = subprocess.Popen(f"type {cmd}", stdout=subprocess.PIPE, shell=True)
    return result.wait() == 0


if __name__ == "__main__":
    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        forward_step,
        args_defaults={"tokenizer_type": "GPT2BPETokenizer"},
    )
    # 本项目的所有注释都依据 scripts/pretrain_codegeex.sh 的运行结果
    # 依据上述脚本运行时所有显式指定与未显式指定的参数值如下:
    # ------------------------ arguments ------------------------
    #   accumulate_allreduce_grads_in_fp32 .............. False
    #   adam_beta1 ...................................... 0.9
    #   adam_beta2 ...................................... 0.95
    #   adam_eps ........................................ 1e-08
    #   adlr_autoresume ................................. False
    #   adlr_autoresume_interval ........................ 1000
    #   apply_query_key_layer_scaling ................... False
    #   apply_residual_connection_post_layernorm ........ False
    #   attention_dropout ............................... 0.1
    #   attention_softmax_in_fp32 ....................... True
    #   beam_search ..................................... False
    #   beam_search_nucleus ............................. False
    #   beam_warmup ..................................... False
    #   beam_warmup_length .............................. 0
    #   bert_binary_head ................................ True
    #   bert_load ....................................... None
    #   bf16 ............................................ False
    #   bias_dropout_fusion ............................. True
    #   bias_gelu_fusion ................................ True
    #   biencoder_projection_dim ........................ 0
    #   biencoder_shared_query_context_model ............ False
    #   block_data_path ................................. None
    #   checkpoint_activations .......................... True
    #   checkpoint_in_cpu ............................... False
    #   checkpoint_num_layers ........................... 1
    #   clip_grad ....................................... 1.0
    #   co_evaluation ................................... False
    #   compress ........................................ False
    #   consumed_train_samples .......................... 0
    #   consumed_train_tokens ........................... 0
    #   consumed_valid_samples .......................... 0
    #   contigious_checkpointing ........................ False
    #   cpu_optimizer ................................... False
    #   cpu_torch_adam .................................. False
    #   data_impl ....................................... mmap
    #   data_parallel_size .............................. 2
    #   data_path ....................................... ['/home/icksys/csw/CodeGeeX/pt_data/my_data']
    #   dataloader_type ................................. single
    #   DDP_impl ........................................ local
    #   decoder_seq_length .............................. None
    #   deepscale ....................................... False
    #   deepscale_config ................................ None
    #   deepspeed ....................................... True
    #   deepspeed_activation_checkpointing .............. True
    #   deepspeed_config ................................ /home/icksys/csw/CodeGeeX/scripts/ds_config.json
    #   deepspeed_mpi ................................... False
    #   dist_timeout .................................... 30
    #   distribute_checkpointed_activations ............. False
    #   distributed_backend ............................. nccl
    #   ds_pipeline_enabled ............................. False
    #   embedding_path .................................. None
    #   encoder_seq_length .............................. 512
    #   eod_mask_loss ................................... False
    #   eval_interval ................................... 10
    #   eval_iters ...................................... 10
    #   evaluation ...................................... False
    #   evidence_data_path .............................. None
    #   exit_duration_in_mins ........................... None
    #   exit_interval ................................... None
    #   ffn_hidden_size ................................. 20480
    #   finetune ........................................ False
    #   force_default ................................... False
    #   force_device .................................... None
    #   fp16 ............................................ True
    #   fp16_lm_cross_entropy ........................... False
    #   fp32_residual_connection ........................ False
    #   global_batch_size ............................... 4
    #   gold ............................................ False
    #   gold_beta ....................................... 0.05
    #   hidden_dropout .................................. 0.1
    #   hidden_size ..................................... 5120
    #   hysteresis ...................................... 2
    #   ict_head_size ................................... None
    #   ict_load ........................................ None
    #   img_dim ......................................... 224
    #   index_cache_dir ................................. None
    #   indexer_batch_size .............................. 128
    #   indexer_log_interval ............................ 1000
    #   init_method_std ................................. 0.02
    #   init_method_xavier_uniform ...................... False
    #   initial_loss_scale .............................. 4294967296
    #   kv_channels ..................................... 128
    #   layernorm_epsilon ............................... 1e-05
    #   lazy_mpu_init ................................... None
    #   ln_fp16 ......................................... True
    #   load ............................................ None
    #   load_state ...................................... /home/icksys/csw/CodeGeeX/scripts/mp4_parallel_weights/
    #   local_rank ...................................... 0
    #   log_batch_size_to_tensorboard ................... False
    #   log_interval .................................... 1
    #   log_learning_rate_to_tensorboard ................ True
    #   log_loss_scale_to_tensorboard ................... True
    #   log_num_zeros_in_grad ........................... False
    #   log_params_norm ................................. False
    #   log_timers_to_tensorboard ....................... False
    #   log_validation_ppl_to_tensorboard ............... False
    #   loss_scale ...................................... 12.0
    #   loss_scale_window ............................... 1000
    #   low_memory_load ................................. None
    #   lr .............................................. 0.0002
    #   lr_decay_iters .................................. 100000
    #   lr_decay_samples ................................ None
    #   lr_decay_style .................................. cosine
    #   lr_decay_tokens ................................. None
    #   lr_warmup_fraction .............................. None
    #   lr_warmup_iters ................................. 1500
    #   lr_warmup_samples ............................... 0
    #   make_vocab_size_divisible_by .................... 52224
    #   mask_prob ....................................... 0.15
    #   masked_softmax_fusion ........................... True
    #   max_position_embeddings ......................... 2048
    #   memory_centric_tiled_linear ..................... False
    #   merge_file ...................................... /home/icksys/csw/CodeGeeX/codegeex/tokenizer/merges.txt
    #   micro_batch_size ................................ 2
    #   min_loss_scale .................................. 1.0
    #   min_lr .......................................... 1e-07
    #   mmap_warmup ..................................... False
    #   ms_model ........................................ False
    #   no_learned_position_embeddings .................. False
    #   no_load_optim ................................... None
    #   no_load_rng ..................................... None
    #   no_pipeline_parallel ............................ True
    #   no_save_optim ................................... None
    #   no_save_rng ..................................... None
    #   num_attention_heads ............................. 40
    #   num_beams ....................................... 4
    #   num_channels .................................... 3
    #   num_classes ..................................... 1000
    #   num_layers ...................................... 39
    #   num_layers_per_virtual_pipeline_stage ........... None
    #   num_workers ..................................... 2
    #   onnx_safe ....................................... None
    #   openai_gelu ..................................... False
    #   optimizer ....................................... adam
    #   override_lr_scheduler ........................... True
    #   params_dtype .................................... torch.float16
    #   partition_activations ........................... False
    #   patch_dim ....................................... 16
    #   pipeline_model_parallel_size .................... 1
    #   play_tau ........................................ 2.0
    #   profile_backward ................................ False
    #   query_in_block_prob ............................. 0.1
    #   rampup_batch_size ............................... None
    #   rank ............................................ 0
    #   remote_device ................................... none
    #   reset_attention_mask ............................ False
    #   reset_position_ids .............................. False
    #   retriever_report_topk_accuracies ................ []
    #   retriever_score_scaling ......................... False
    #   retriever_seq_length ............................ 256
    #   reward_growth ................................... constant
    #   sample_rate ..................................... 1.0
    #   save ............................................ /home/icksys/csw/CodeGeeX/scripts/pretrain-codegeex-13b-test
    #   save_interval ................................... 100
    #   scale_embeddings ................................ False
    #   scaled_upper_triang_masked_softmax_fusion ....... False
    #   scatter_gather_tensors_in_pipeline .............. True
    #   scattered_embeddings ............................ False
    #   seed ............................................ 1234
    #   seq_length ...................................... 512
    #   sgd_momentum .................................... 0.9
    #   short_seq_prob .................................. 0.1
    #   shrink_embedding_gradient_alpha ................. 1.0
    #   shrink_embedding_gradient_steps ................. None
    #   shrink_logit_embedding_gradient ................. False
    #   split ........................................... 100,0,0
    #   split_transformers .............................. False
    #   synchronize_each_layer .......................... False
    #   tempering ....................................... None
    #   tensor_model_parallel_size ...................... 4
    #   tensorboard_dir ................................. /home/icksys/csw/CodeGeeX/scripts/pretrain-codegeex-13b-test/tb20240202_103349
    #   tensorboard_log_interval ........................ 1
    #   tensorboard_queue_size .......................... 1000
    #   test_data_path .................................. None
    #   tile_factor ..................................... 1
    #   titles_data_path ................................ None
    #   tokenizer_path .................................. None
    #   tokenizer_type .................................. GPT2BPETokenizer
    #   train_iters ..................................... 25
    #   train_samples ................................... None
    #   train_tokens .................................... None
    #   use_checkpoint_lr_scheduler ..................... False
    #   use_contiguous_buffers_in_ddp ................... False
    #   use_cpu_initialization .......................... None
    #   use_one_sent_docs ............................... False
    #   use_pin_memory .................................. False
    #   valid_data_path ................................. None
    #   virtual_pipeline_model_parallel_size ............ None
    #   vocab_extra_ids ................................. 0
    #   vocab_file ...................................... /home/icksys/csw/CodeGeeX/codegeex/tokenizer/vocab.json
    #   wandb_log_interval .............................. 1
    #   wandb_logging ................................... False
    #   weight_decay .................................... 0.1
    #   world_size ...................................... 8
    #   zero_allgather_bucket_size ...................... 0.0
    #   zero_contigious_gradients ....................... False
    #   zero_reduce_bucket_size ......................... 0.0
    #   zero_reduce_scatter ............................. False
    #   zero_stage ...................................... 2
    # -------------------- end of arguments ---------------------
