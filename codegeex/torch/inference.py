import copy
import json
import os
import time
from typing import *

import torch
import torch.nn.functional as F
from dataclasses import dataclass


def get_ltor_masks_and_position_ids(
    data, 
    eod_token, 
    reset_position_ids, 
    reset_attention_mask, 
):
    """Build masks and position id for left to right model."""

    # Extract batch size and sequence length.
    micro_batch_size, seq_length = data.size()

    # Attention mask (lower triangular).
    if reset_attention_mask:
        att_mask_batch = micro_batch_size
    else:
        att_mask_batch = 1
    attention_mask = torch.tril(
        torch.ones((att_mask_batch, seq_length, seq_length), device=data.device)
    ).view(att_mask_batch, 1, seq_length, seq_length)

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    position_ids = position_ids.unsqueeze(0).expand_as(data)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        position_ids = position_ids.clone()

    if reset_position_ids or reset_attention_mask:
        # Loop through the batches:
        for b in range(micro_batch_size):

            # Find indecies where EOD token is.
            eod_index = position_ids[b, data[b] == eod_token]
            # Detach indecies from positions if going to modify positions.
            if reset_position_ids:
                eod_index = eod_index.clone()

            # Loop through EOD indecies:
            prev_index = 0
            for j in range(eod_index.size()[0]):
                i = eod_index[j]
                # Mask attention loss.
                if reset_attention_mask:
                    attention_mask[b, 0, (i + 1) :, : (i + 1)] = 0
                # Reset positions.
                if reset_position_ids:
                    position_ids[b, (i + 1) :] -= i + 1 - prev_index
                    prev_index = i + 1

    # Convert attention mask to binary:
    attention_mask = attention_mask < 0.5

    return attention_mask, position_ids


def get_batch(
    context_tokens, 
    micro_batch_size, 
    eod_token, 
    reset_position_ids=False,
    reset_attention_mask=False,
):
    """Generate batch from context tokens."""
    tokens = context_tokens.view(micro_batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_token,
        reset_position_ids,
        reset_attention_mask,
    )

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    """This function has been mostly taken from huggingface conversational
    ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-
             conversational-ai-with-transfer-learning-2d818ac26313"""

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def pad_batch(batch, pad_id, seq_length):
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < seq_length:
            tokens.extend([pad_id] * (seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def forward_step(
        model,
        tokens,
        seq_length,
        position_ids,
        attention_mask,
        layer_past=None,
        get_key_value=None,
        prompt_length=None,
        context_length=None,
):
    # Forward pass through the model.
    output_tensor = model(
        tokens,
        position_ids,
        attention_mask,
        layer_past=layer_past,
        get_key_value=get_key_value,
        prompt_length=prompt_length,
        context_length=context_length,
    )

    if get_key_value:
        output_tensor, layer_past = output_tensor

    if get_key_value:
        return output_tensor, layer_past

    return output_tensor


def get_token_stream(
        model,
        tokenizer,
        seq_length,
        # seq_length: 2048
        out_seq_length,
        # out_seq_length: 1024
        context_tokens,
        return_scores: bool = False,
        prompt_length: int = None,
        micro_batch_size: int = None,
        # micro_batch_size: 1
        bad_ids: List = None,
        # bad_ids: None
        temperature: float = 1.0,
        # temperature: 0.8
        topp: float = 1.0,
        # topp: 0.95
        topk: int = 0.0,
        # topk: 0
        greedy: bool = False,
        # greedy: False
        recompute: bool = False,
        # recompute: False
):
    context_tokens, context_lengths = pad_batch(context_tokens, tokenizer.eos_token_id, seq_length)
    # tokenizer.eos_token_id: 50256
    # context_tokens before pad:
    # [[8189, 11059, 198, 29584, 25, 198, 11377, 1398, 28186, 1391, 198, 50268, 11377, 9037, 25131, 468, 26125, 36,
    #     3639, 7, 600, 21737, 997, 82, 11, 493, 11387, 8, 1391, 198, 50272, 1640, 357, 600, 1312, 796, 657, 26, 1312,
    #     1279, 997, 82, 13, 13664, 532, 352, 26, 1312, 29577, 1391, 198, 50274, 50266, 1640, 357, 600, 474, 796, 1312,
    #     1343, 352, 26, 474, 1279, 997, 82, 13, 13664, 26, 474, 29577, 1391, 198, 50274, 50270, 361, 357, 37372, 13,
    #     8937, 7, 77, 5700, 58, 72, 60, 532, 997, 82, 58, 73, 12962, 1279, 11387, 8, 1391, 198, 50274, 50274, 7783, 2081,
    #     26, 198, 50274, 50270, 92, 198, 50274, 50266, 92, 198, 50272, 92, 198, 50272, 7783, 3991, 26, 198, 50268, 92,
    #     198, 92, 198, 37906, 25]]
    # context_tokens after pad: 用50256 padding到2048长度
    # context_lengths: [126]

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    # context_tokens_tensor.shape: [1, 2048]
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    # context_length_tensor.shape: [1]
    context_length = context_length_tensor.min().item()
    # context_length: 126

    tokens, attention_mask, position_ids = get_batch(
        context_tokens_tensor, 
        micro_batch_size,
        tokenizer.eos_token_id,
    )
    # tokens即为context_tokens_tensor
    # tokens.shape: [1, 2048]

    # attention_mask.shape: [1, 1, 2048, 2048]
    # attention_mask[0, 0, :10, :10]:
    # tensor([[False,  True,  True,  True,  True,  True,  True,  True,  True,  True],
    #         [False, False,  True,  True,  True,  True,  True,  True,  True,  True],
    #         [False, False, False,  True,  True,  True,  True,  True,  True,  True],
    #         [False, False, False, False,  True,  True,  True,  True,  True,  True],
    #         [False, False, False, False, False,  True,  True,  True,  True,  True],
    #         [False, False, False, False, False, False,  True,  True,  True,  True],
    #         [False, False, False, False, False, False, False,  True,  True,  True],
    #         [False, False, False, False, False, False, False, False,  True,  True],
    #         [False, False, False, False, False, False, False, False, False,  True],
    #         [False, False, False, False, False, False, False, False, False, False]],
    #        device='cuda:0')
    #
    # position_ids.shape: [1, 2048]
    # position_ids: tensor([[   0,    1,    2,  ..., 2045, 2046, 2047]], device='cuda:0')

    batch_token_iterator = sample_sequence_batch(
        model,
        tokenizer,
        context_tokens_tensor,
        context_length_tensor,
        attention_mask,
        position_ids,
        seq_length=seq_length,
        out_seq_length=out_seq_length,
        return_scores=return_scores,
        prompt_length=prompt_length,
        bad_ids=bad_ids,
        temperature=temperature,
        topp=topp,
        topk=topk,
        greedy=greedy,
        recompute=recompute,
    )

    for tokens, lengths in batch_token_iterator:
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], lengths
        else:
            yield None, None


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
        model,
        tokenizer,
        context_tokens,
        # context_tokens.shape: [1, 2048]
        context_lengths,
        # context_lengths.shape: [1]
        attention_mask,
        # attention_mask.shape: [1, 1, 2048, 2048]
        position_ids,
        # position_ids.shape: [1, 2048]
        seq_length,
        # seq_length: 2048
        out_seq_length,
        # out_seq_length: 1024
        maxlen=None,
        # maxlen: None
        return_scores: bool = False,
        # return_scores: False
        prompt_length: int = None,
        # prompt_length: None
        bad_ids: List = None,
        # bad_ids: None
        temperature: float = 1.0,
        # temperature: 0.8
        topp: float = 1.0,
        # topp: 0.95
        topk: int = 0.0,
        # topk: 0
        recompute: bool = False,
        # recompute: False
        greedy: bool = False,
        # greedy: False
):
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        # context_length: 126
        eos_id = tokenizer.eos_token_id
        # eos_id: 50256

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        # maxlen: None
        if maxlen is None:
            maxlen = seq_length - 1
            # maxlen: 2047
            if maxlen > (org_context_length + out_seq_length):
                maxlen = org_context_length + out_seq_length
                # org_context_length: 126
                # out_seq_length: 1024
                # maxlen: 1150

        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        # lengths: tensor([1150], device='cuda:0')

        # return_scores: False
        if return_scores:
            scores = torch.zeros([batch_size]).float().cuda()

        while context_length <= (maxlen):

            # recompute: False
            if recompute:
                logits = model(tokens,
                               position_ids,
                               attention_mask,
                               prompt_length=prompt_length,
                               context_length=context_length,
                               )
                logits = logits[:, context_length - 1, :]
            else:
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(
                        batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(
                        batch_size, -1)
                logits, layer_past = model(tokens2use,
                                           positions2use,
                                           attention_mask,
                                           layer_past=layer_past,
                                           get_key_value=True,
                                           prompt_length=prompt_length,
                                           context_length=context_length,
                                           )
                # 第一次进入时:
                # tokens2use.shape: [1, 126]
                # positions2use.shape: [1, 126]
                # attention_mask.shape: [1, 1, 2048, 2048]
                # layer_past: None
                # get_key_value: True
                # prompt_length: None
                # context_length: 126
                # logits.shape: [1, 126, 52224]

                # 第二次进入时:
                # tokens2use: tensor([[198]], device='cuda:0')
                # positions2use: tensor([[126]], device='cuda:0')
                # attention_mask.shape: [1, 1, 2048, 2048]
                # layer_past: 上一轮的layer_past
                # get_key_value: True
                # prompt_length: None
                # context_length: 127
                # logits.shape: [1, 1, 52224]

                logits = logits[:, -1].view(batch_size, -1).contiguous()

            # bad_ids: None
            if bad_ids is not None:
                for bad_id in bad_ids:
                    logits[:, bad_id] = -10000
            # greedy: False
            if greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                # return_scores: False
                if return_scores:
                    orig_log_probs = torch.log_softmax(logits, dim=-1)
                logits /= temperature
                logits = top_k_logits(logits, top_k=topk, top_p=topp)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)
                # 第一次:
                # prev: tensor([198], device='cuda:0')
                # 第二次:
                # prev: tensor([50268], device='cuda:0')

            started = context_lengths <= context_length
            # context_lengths: tensor([126], device='cuda:0')
            # 第一次:
            # context_length: 126
            # started: tensor([True], device='cuda:0')
            # 第二次:
            # context_length: 127
            # started: tensor([True], device='cuda:0')

            new_tokens = switch(tokens[:, context_length].view(-1), prev, started)
            # tokens.shape: [1, 2048]
            # 第一次:
            # tokens[:, context_length].view(-1): tensor([50256], device='cuda:0')
            # new_tokens: tensor([198], device='cuda:0')
            # 第二次:
            # tokens[:, context_length].view(-1): tensor([50256], device='cuda:0')
            # new_tokens: tensor([50268], device='cuda:0')

            if not greedy and return_scores:
                indices = prev.view(-1, 1)
                new_scores = orig_log_probs.gather(1, indices).view(-1)
                new_scores = new_scores * started
                new_scores = new_scores * is_done.bool().logical_not()
                scores += new_scores

            tokens[:, context_length] = new_tokens
            done_token = (prev == eos_id).byte() & started.byte()
            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token
            done = torch.all(is_done)
            # 第一次:
            # prev == eos_id: tensor([False], device='cuda:0')
            # (prev == eos_id).byte(): tensor([0], device='cuda:0', dtype=torch.uint8)
            # started.byte(): tensor([1], device='cuda:0', dtype=torch.uint8)
            # done_token: tensor([0], device='cuda:0', dtype=torch.uint8)
            # is_done: tensor([0], device='cuda:0', dtype=torch.uint8)
            # just_finished: tensor([False], device='cuda:0')
            # lengths: tensor([1150], device='cuda:0')
            # is_done: tensor([0], device='cuda:0', dtype=torch.uint8)
            # done: tensor(0, device='cuda:0', dtype=torch.uint8)

            # 第二次:
            # done_token: tensor([0], device='cuda:0', dtype=torch.uint8)
            # just_finished: tensor([False], device='cuda:0')
            # lengths: tensor([1150], device='cuda:0')
            # is_done: tensor([0], device='cuda:0', dtype=torch.uint8)
            # done: tensor(0, device='cuda:0', dtype=torch.uint8)
            
            if return_scores:
                yield tokens, (lengths, scores)
            else:
                yield tokens, lengths
                # tokens.shape: [1, 2048]
                
            context_length += 1
            counter += 1
            if done:
                break
