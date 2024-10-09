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

"""Transformer based language model."""

import torch
import torch.nn.functional as F

from codegeex.megatron import get_args
from codegeex.megatron import mpu, print_rank_0
from codegeex.megatron.model.module import MegatronModule
from codegeex.megatron.model.transformer import ParallelTransformer
from codegeex.megatron.model.utils import init_method_normal, scaled_init_method_normal
from codegeex.megatron.mpu.initialize import get_tensor_model_parallel_world_size


def get_shrink_embedding_gradient_alpha(iteration):
    args = get_args()

    alpha = args.shrink_embedding_gradient_alpha
    if args.shrink_embedding_gradient_steps is None:
        return alpha
    else:
        x1 = int(args.shrink_embedding_gradient_steps[0])
        x2 = int(args.shrink_embedding_gradient_steps[1])
        if iteration <= x1:
            return alpha
        elif iteration >= x1 + x2:
            return 1.0
        else:
            return alpha + (1 - alpha) * (args.iteration - x1) / x2
        

def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""

    # input_.shape: [b, s, h], dtype: torch.float16
    # word_embeddings_weight.shape: [vocab_size/p, h], dtype: torch.float16
    # parallel_output: True
    # bias: None

    # Parallel logits.
    input_parallel = mpu.copy_to_tensor_model_parallel_region(input_)
    # input_parallel.shape: [b, s, h], dtype: torch.float16
    # Matrix multiply.
    args = get_args()
    # args.shrink_logit_embedding_gradient: False
    if args.shrink_logit_embedding_gradient:
        if hasattr(args, 'iteration'):
            alpha = get_shrink_embedding_gradient_alpha(args.iteration + 1)
        else:
            alpha = args.shrink_embedding_gradient_alpha
        word_embeddings_weight = word_embeddings_weight if alpha == 1.0 \
            else (
                word_embeddings_weight * alpha +
                word_embeddings_weight.detach() * (1 - alpha)
        )
    # word_embeddings_weight.shape: [vocab_size/p, h]
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight.half())
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight.half(), bias)
    # logits_parallel.shape: [b, s, vocab_size/p], dtype: torch.float16
    # Gather if needed.
    # parallel_output: True
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_tensor_model_parallel_region(logits_parallel)


def get_language_model(
        num_tokentypes,
        add_pooler,
        init_method=None,
        scaled_init_method=None,
):
    """Build language model and return along with the key to save."""

    # num_tokentypes: 0
    # add_pooler: False
    # init_method: init_method_normal(0.02)
    # scaled_init_method: scaled_init_method_normal(0.02, 39)

    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    # Language model.
    language_model = TransformerLanguageModel(
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        num_tokentypes=num_tokentypes,
        add_pooler=add_pooler)
    # key used for checkpoints.
    language_model_key = 'language_model'

    return language_model, language_model_key


class Embedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
    ):

        # hidden_size: 5120, 每个 token 的向量维度
        # vocab_size: 52224, 词表大小
        # max_sequence_length: 2048, 最大上下文长度
        # embedding_dropout_prob: 0.1, dropout probability for embeddings
        # init_method: init_method_normal(0.02), 初始化权重的方法
        # num_tokentypes: 0, 类似于 Bert 中的 segment type

        super(Embedding, self).__init__()
        
        args = get_args()
        
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.max_sequence_length = max_sequence_length
        
        # Word embeddings (parallel).
        # self.word_embeddings.weight.shape: [vocab_size//p, h], dtype: torch.float16
        # p 表示 TP 组模型并行度
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, self.hidden_size, init_method=self.init_method)
        self._word_embeddings_key = 'word_embeddings'
            
        self.vocab_size = vocab_size

        # Position embedding (serial).
        # self.position_embeddings.weight.shape: [max_sequence_length, h]
        self.position_embeddings = torch.nn.Embedding(
            max_sequence_length, self.hidden_size)
        self.position_embeddings = self.position_embeddings.half()
        self._position_embeddings_key = 'position_embeddings'
            
        # Initialize the position embeddings.
        self.init_method(self.position_embeddings.weight)
        # self.position_embeddings.weight.dtype: torch.float16

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        # tokentype_embeddings类似于 Bert 中的 segment embedding
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        # 使用的是常驻随机数种子
        # embedding_dropout_prob: 0.1
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        # 如果在 pretrain 阶段未定义 TE，而在 fine-tune 阶段 TE，则可通过此函数添加
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # ===============================分布式推断时进入该函数的参数================================
        # 以第一次推断时输入 tokens 的长度 126 为例
        #                          首次推断                     下次推断
        # input_ids.shape:         [1, 126]                    [1, 1]
        # position_ids:            [1, 126]                    [1, 1]
        # 其余参数为缺省值
        #
        # embeddings.shape:        [1, 126, h]                 [1, 1, h]
        # ======================================================================================

        # ===============================分布式训练时进入该函数的参数================================
        # input_ids.shape: [b, s], dtype: torch.int64
        # position_ids.shape: [b, s], dtype: torch.int64
        # 训练时其余参数为缺省值
        # ======================================================================================

        """定义输入过 embedding 层的计算方法"""
        # Embeddings.
        # words_embeddings.shape: [b, s, h], dtype: torch.float16
        # 再次注意: self.word_embeddings 做 forward 时, 最终的输出结果是 AllReduce 的
        words_embeddings = self.word_embeddings(input_ids)
        # position_embeddings.shape: [b, s, h], dtype: torch.float16
        position_embeddings = self.position_embeddings(position_ids)
        # embeddings.shape: [b, s, h], dtype: torch.float16
        embeddings = words_embeddings + position_embeddings
        # 依需要决定是否增加 TE
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        # 使用的是常驻随机数种子
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix='', keep_vars=False,
    ):
        """For easy load."""
        state_dict_ = {}
        state_dict_[self._word_embeddings_key] \
            = self.word_embeddings.state_dict(destination, prefix, keep_vars)
        state_dict_[self._position_embeddings_key] \
            = self.position_embeddings.state_dict(
            destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        # 用于模型的重载, 例如训到一半挂掉了, 我们就重新初始化一个新模型, 重载上个 checkpoint 保存下的权重。

        # Word embedding.
        if self._word_embeddings_key in state_dict:
            state_dict_ = state_dict[self._word_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'word_embeddings' in key:
                    state_dict_[key.split('word_embeddings.')[1]] \
                        = state_dict[key]
        vocab_len = state_dict_['weight'].shape[0]
        state_dict_["weight"] = state_dict_["weight"][:self.vocab_size // get_tensor_model_parallel_world_size()]
        self.word_embeddings.load_state_dict(state_dict_, strict=strict)

        # Position embedding.
        if self._position_embeddings_key in state_dict:
            state_dict_ = state_dict[self._position_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'position_embeddings' in key:
                    state_dict_[key.split('position_embeddings.')[1]] \
                        = state_dict[key]
        
        pos_len = state_dict_['weight'].shape[0]
        max_seq_len = self.max_sequence_length
        if pos_len < max_seq_len:
            print_rank_0(f"Position embedding padded {pos_len} -> {max_seq_len}.")
            position_embeddings_padded = torch.nn.Embedding(
            max_seq_len - pos_len, self.hidden_size).half()
            self.init_method(position_embeddings_padded.weight)
            state_dict_['weight'] = torch.cat([state_dict_['weight'], position_embeddings_padded.weight], dim=0)

        # self.position_embeddings = self.position_embeddings.half()
        self.position_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class EmbeddingPipe(Embedding):
    def forward(self, inputs, **kwargs):
        if not hasattr(self, "_args"):
            self._args = get_args()

        input_ids = inputs[0]
        position_ids = inputs[1]
        if hasattr(self._args, "attn_mask"):
            attention_mask = None
        else:
            attention_mask = inputs[2]

        if len(inputs) == 4:
            tokentype_ids = inputs[3]
        else:
            tokentype_ids = None

        embeddings = super().forward(
            input_ids, position_ids, tokentype_ids=tokentype_ids
        )

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, "attn_mask"):
            return embeddings
        else:
            assert False
            return embeddings, attention_mask

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.word_embeddings.weight


class QueryEmbedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0):
        super(QueryEmbedding, self).__init__()

        # hidden_size: 5120, 每个 token 的向量维度
        # vocab_size: 52224, 词表大小
        # max_sequence_length: 2048, 最大上下文长度
        # embedding_dropout_prob: 0.1,  dropout probability for embeddings
        # init_method: init_method_normal(0.02), 初始化权重的方法
        # num_tokentypes: 0, 类似于 Bert 中的 segment type

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.max_sequence_length = max_sequence_length
        
        # Top query position embedding (serial).
        self.top_query_embeddings = mpu.VocabParallelEmbedding(
            max_sequence_length, self.hidden_size, init_method=self.init_method)
        self.top_query_embeddings = self.top_query_embeddings.half()
        self._top_query_embeddings_key = 'top_query_embeddings'
            
        # Initialize the top query position embeddings.
        self.init_method(self.top_query_embeddings.weight)
        # self.top_query_embeddings.weight.shape: [max_sequence_length//p, h], dtype: torch.float16

        # Token type embedding.
        # Add this as an optional field that can be added through
        # method call so we can load a pretrain model without
        # token types and add them as needed.
        self._tokentype_embeddings_key = 'tokentype_embeddings'
        if self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes,
                                                           self.hidden_size)
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        # 使用的是常驻随机数种子
        # embedding_dropout_prob: 0.1
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes),
                  flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, position_ids, tokentype_ids=None):
        # ===============================分布式推断时进入该函数的参数================================
        # 以第一次推断时输入 tokens 的长度 126 为例
        #                          首次推断                     下次推断
        # position_ids:            [1, 126]                    [1, 1]
        # 其余参数为缺省值
        #
        # embeddings.shape:        [1, 126, h]                 [1, 1, h]
        # ======================================================================================

        # ===============================分布式训练时进入该函数的参数================================
        # position_ids.shape: [b, s], dtype: torch.int64
        # 训练时其余参数为缺省值
        # ======================================================================================

        # Embeddings.

        embeddings = self.top_query_embeddings(position_ids)
        # embeddings.shape: [b, s, h], dtype: torch.float16
        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        # Dropout.
        # 使用的是常驻随机数种子
        embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._top_query_embeddings_key] \
            = self.top_query_embeddings.state_dict(
            destination, prefix, keep_vars)
        if self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] \
                = self.tokentype_embeddings.state_dict(
                destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Position embedding.
        if self._top_query_embeddings_key in state_dict:
            state_dict_ = state_dict[self._top_query_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'top_query_embeddings' in key:
                    state_dict_[key.split('top_query_embeddings.')[1]] \
                        = state_dict[key]
        pos_len = state_dict_['weight'].shape[0]
        max_seq_len = self.max_sequence_length // get_tensor_model_parallel_world_size()
        if pos_len < max_seq_len:
            print_rank_0(f"Top query embedding padded {pos_len} -> {max_seq_len}.")
            top_query_embeddings_padded = torch.nn.Embedding(
            max_seq_len - pos_len, self.hidden_size).half()
            self.init_method(top_query_embeddings_padded.weight)
            state_dict_['weight'] = torch.cat([state_dict_['weight'], top_query_embeddings_padded.weight], dim=0)
        self.top_query_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tokentype_embeddings' in key:
                        state_dict_[key.split('tokentype_embeddings.')[1]] \
                            = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_,
                                                          strict=strict)
            else:
                print('***WARNING*** expected tokentype embeddings in the '
                      'checkpoint but could not find it', flush=True)


class QueryEmbeddingPipe(QueryEmbedding):
    def forward(self, inputs, **kwargs):
        if not hasattr(self, "_args"):
            self._args = get_args()

        position_ids = inputs[0]
        if hasattr(self._args, "attn_mask"):
            attention_mask = None
        else:
            attention_mask = inputs[1]

        if len(inputs) == 3:
            tokentype_ids = inputs[2]
        else:
            tokentype_ids = None

        embeddings = super().forward(
            position_ids, tokentype_ids=tokentype_ids,
        )

        # If cmd args has attn_mask, we don't forward it as an activation.
        if hasattr(self._args, "attn_mask"):
            return embeddings
        else:
            assert False
            return embeddings, attention_mask

    @property
    def word_embeddings_weight(self):
        """Easy accessory for the DeepSpeed pipeline engine to tie embeddings across stages."""
        return self.top_query_embeddings.weight
    
    
class TransformerLanguageModel(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
          masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False):

        # init_method: init_method_normal(0.02)
        # output_layer_init_method: scaled_init_method_normal(0.02, 39)
        # num_tokentypes: 0
        # add_pooler: False

        super(TransformerLanguageModel, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        # num_tokentypes: 0
        self.init_method = init_method
        self.add_pooler = add_pooler
        # add_pooler: False

        # Embeddings
        self.embedding = Embedding(self.hidden_size,
                                   args.padded_vocab_size,
                                   args.max_position_embeddings,
                                   args.hidden_dropout,
                                   self.init_method,
                                   self.num_tokentypes)
        self._embedding_key = 'embedding'

        # Query embeddings
        self.topQueryEmbedding = QueryEmbedding(self.hidden_size,
                                                args.padded_vocab_size,
                                                args.max_position_embeddings,
                                                args.hidden_dropout,
                                                self.init_method,
                                                self.num_tokentypes)
        self._topQueryEmbedding_key = 'topQueryEmbedding'

        # Transformer
        self.transformer = ParallelTransformer(
            self.init_method,
            output_layer_init_method)
        self._transformer_key = 'transformer'

    def set_input_tensor(self, input_tensor):
        """See megatron.model.transformer.set_input_tensor()"""
        self.transformer.set_input_tensor(input_tensor)
        
    def forward(
            self,
            input_ids,
            position_ids,
            attention_mask,
            tokentype_ids=None,
            layer_past=None,
            get_key_value=False,
            pooling_sequence_index=0,
            prompt_length=None,
            context_length=None,
    ):
        # ===============================分布式推断时进入该函数的参数================================
        # 以第一次推断时输入 tokens 的长度 126 为例
        #                          首次推断                     下次推断
        # input_ids.shape:         [1, 126]                    [1, 1]
        # position_ids:            [1, 126]                    [1, 1]
        # attention_mask:          [1, 1, 2048, 2048]          [1, 1, 2048, 2048]
        # layer_past:              None                        首次推断返回的layer_past
        # get_key_value:           True                        True
        # prompt_length:           None                        None
        # context_length:          126                         127
        # 其余参数为缺省值
        #
        # transformer_output[0].shape:[1, 126, h]                 [1, 1, h]
        # ======================================================================================

        # ===============================分布式训练时进入该函数的参数================================
        # input_ids.shape: [b, s], dtype: torch.int64
        # position_ids.shape: [b, s], dtype: torch.int64
        # attention_mask.shape: [1, 1, s, s], dtype: torch.bool
        # 训练时其余参数为缺省值
        # ======================================================================================

        # Embeddings.
        embedding_output = self.embedding(input_ids, position_ids,
                                          tokentype_ids=tokentype_ids)
        # embedding_output.shape: [b, s, h], dtype: torch.float16
        query_position_ids = position_ids
        queryEmbedding_out = self.topQueryEmbedding(query_position_ids,
                                                    tokentype_ids=tokentype_ids)
        # queryEmbedding_out.shape: [b, s, h], dtype: torch.float16

        # Transformer.
        transformer_output = self.transformer(embedding_output,
                                              queryEmbedding_out,
                                              attention_mask,
                                              layer_past=layer_past,
                                              get_key_value=get_key_value,
                                              prompt_length=prompt_length,
                                              context_length=context_length, )
        # transformer_output.shape: [b, s, h], dtype: torch.float16

        return transformer_output

    def state_dict_for_save_checkpoint(self, destination=None, prefix='',
                                       keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        state_dict_[self._embedding_key] \
            = self.embedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        state_dict_[self._topQueryEmbedding_key] \
            = self.topQueryEmbedding.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        state_dict_[self._transformer_key] \
            = self.transformer.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars)
        if self.add_pooler:
            state_dict_[self._pooler_key] \
                = self.pooler.state_dict_for_save_checkpoint(
                destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self._embedding_key in state_dict:
            state_dict_ = state_dict[self._embedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.embedding.load_state_dict(state_dict_, strict=strict)

        if self._topQueryEmbedding_key in state_dict:
            state_dict_ = state_dict[self._topQueryEmbedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.topQueryEmbedding.load_state_dict(state_dict_, strict=strict)

        # Transformer.
        if self._transformer_key in state_dict:
            state_dict_ = state_dict[self._transformer_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]
        self.transformer.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.add_pooler:
            assert 'pooler' in state_dict, \
                'could not find data for pooler in the checkpoint'
            self.pooler.load_state_dict(state_dict[self._pooler_key],
                                        strict=strict)
