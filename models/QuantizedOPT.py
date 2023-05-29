import os
import torch.nn as nn

from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel, OPTDecoder, OPTLearnedPositionalEmbedding, OPTAttention
from transformers.models.opt.modeling_opt import ACT2FN

from models.module import quantized_Linear


class QuantizedOPTAttention(OPTAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        bitW: int = 8,
        bitA: int = 8,
        vector_index: list = [0, 1]
    ):
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            is_decoder,
            bias
        )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = quantized_Linear(embed_dim, embed_dim, bias=bias, bitW=bitW, bitA=bitA, vector_index=vector_index)
        self.v_proj = quantized_Linear(embed_dim, embed_dim, bias=bias, bitW=bitW, bitA=bitA, vector_index=vector_index)
        self.q_proj = quantized_Linear(embed_dim, embed_dim, bias=bias, bitW=bitW, bitA=bitA, vector_index=vector_index)
        self.out_proj = quantized_Linear(embed_dim, embed_dim, bias=bias, bitW=bitW, bitA=bitA, vector_index=vector_index)


class QuantizedOPTDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = QuantizedOPTAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=config.enable_bias,
            bitW=config.bitW, bitA=config.bitA, vector_index=config.vector_index
        )
        self.do_layer_norm_before = config.do_layer_norm_before
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]

        self.self_attn_layer_norm = nn.LayerNorm(
            self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine
        )
        self.fc1 = quantized_Linear(self.embed_dim, config.ffn_dim, bias=config.enable_bias, bitW=config.bitW, bitA=config.bitA, vector_index=config.vector_index)
        self.fc2 = quantized_Linear(config.ffn_dim, self.embed_dim, bias=config.enable_bias, bitW=config.bitW, bitA=config.bitA, vector_index=config.vector_index)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=config.layer_norm_elementwise_affine)


class QuantizedOPTDecoder(OPTDecoder):

    def __init__(self, config):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_out = quantized_Linear(
                config.hidden_size, config.word_embed_proj_dim, bias=False,
                bitW=config.bitW, bitA=config.bitA, vector_index=config.vector_index
            )
        else:
            self.project_out = None

        if config.word_embed_proj_dim != config.hidden_size:
            self.project_in = nn.Linear(config.word_embed_proj_dim, config.hidden_size, bias=False)
        else:
            self.project_in = None

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([QuantizedOPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()


class QuantizedOPTModel(OPTModel):

    def __init__(self, config):
        super().__init__(config)
        self.decoder = QuantizedOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()


class QuantizedOPTForCausalLM(OPTForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        config.bitW = 8
        config.bitA = 8
        config.vector_index = [0, 1]
        self.model = QuantizedOPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()


if __name__ == '__main__':

    import torch
    from transformers import OPTConfig, AutoConfig
    # """
    model_name = 'facebook/opt-125m'
    # config = OPTConfig()
    config = AutoConfig.from_pretrained(model_name)
    config.bitW = 8
    config.bitA = 8
    config.vector_index = [0, 1]
    model = QuantizedOPTForCausalLM(config)
    # """
    """
    model = QuantizedOPTForCausalLM.from_pretrained(model_name, torch_dtype='auto')
    """