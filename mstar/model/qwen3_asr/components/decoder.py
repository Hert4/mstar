"""Qwen3-ASR text decoder: a dense Qwen3 stack on the shared components.

Every piece already existed. A layer is the shared ``DecoderLayer``
(pre-norm, attention then MLP) composed from ``Attention`` with QK-norm
and GQA, ``GatedMLP`` (SwiGLU), and two ``RMSNorm``s — which is exactly
what the checkpoint's 11 tensors per layer describe:

    model.layers.{i}.input_layernorm.weight
    model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    model.layers.{i}.self_attn.{q,k}_norm.weight
    model.layers.{i}.post_attention_layernorm.weight
    model.layers.{i}.mlp.{gate,up,down}_proj.weight

No cross-attention, unlike the Whisper decoder next door: the audio is
already in the token sequence by the time this runs, spliced into the
embeddings at ``audio_token_id`` positions. So one KV cache, one
attention resource, and the audio pays ordinary per-token KV.

RoPE is the plain 1-D kind (see ``config`` for why the checkpoint's
``mrope_section`` does not need the multi-axis path), with Qwen3's
``rope_theta`` of 1e6.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from mstar.model.components.attention import Attention
from mstar.model.components.decoder_layer import DecoderLayer
from mstar.model.components.mlp import GatedMLP
from mstar.model.components.norm import RMSNorm
from mstar.model.qwen3_asr.config import ATTN, KV_CACHE, POS, Qwen3ASRModelConfig


def _layer(config: Qwen3ASRModelConfig) -> DecoderLayer:
    return DecoderLayer(
        self_attn=Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            # Qwen3 states head_dim explicitly; it is not hidden/heads here
            # (1024/16 = 64, but the checkpoint's q_proj is [2048, 1024]).
            head_dim=config.head_dim,
            qkv_bias=False,
            o_bias=False,
            qk_norm=True,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            attn_key=ATTN,
            kv_key=KV_CACHE,
            pos_key=POS,
        ),
        mlp=GatedMLP(config.hidden_size, config.intermediate_size, "silu", bias=False),
        input_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
        post_attention_layernorm=RMSNorm(config.hidden_size, eps=config.rms_norm_eps),
    )


class Qwen3ASRDecoder(nn.Module):
    """``embed_tokens`` → N ``DecoderLayer`` → ``norm``, plus the LM head."""

    def __init__(self, config: Qwen3ASRModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [_layer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # The checkpoint ships ``thinker.lm_head.weight`` even though
        # ``tie_word_embeddings`` is true, so load it rather than aliasing
        # the embedding: if the two ever diverge, aliasing would silently
        # decode against the wrong matrix.
        self.lm_head_weight = nn.Parameter(
            torch.empty(config.vocab_size, config.hidden_size)
        )

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings. Positions are RoPE'd inside attention, so
        unlike Whisper there is nothing to add here."""
        return self.embed_tokens(input_ids)

    def splice_audio(
        self, input_embeds: torch.Tensor, input_ids: torch.Tensor,
        audio_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Write ``audio_embeds`` over the ``audio_token_id`` placeholders.

        The prompt carries exactly one placeholder per encoder output
        token (``Qwen3ASRModelConfig.audio_output_len``), so this is a
        positional overwrite, not an insertion — the sequence length is
        already right. A count mismatch means the prompt and the encoder
        disagree about the audio, which would silently shift every later
        token, so it raises instead.
        """
        mask = input_ids == self.config.audio_token_id
        slots = int(mask.sum())
        if slots != audio_embeds.shape[0]:
            raise ValueError(
                f"prompt has {slots} audio placeholders but the encoder "
                f"produced {audio_embeds.shape[0]} tokens"
            )
        if slots == 0:
            return input_embeds
        out = input_embeds.clone()
        out[mask] = audio_embeds.to(dtype=out.dtype)
        return out

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.lm_head_weight)

    def forward(self, input_embeds: torch.Tensor, *, label: str) -> torch.Tensor:
        hidden_states = input_embeds
        # Label and layer index are cursors on the shared resources: bind the
        # label once, advance the index per layer. Passing them per call
        # would have inductor specialize on the int.
        self.layers[0].self_attn.attend.bind_step(label)
        for layer_idx, layer in enumerate(self.layers):
            layer.self_attn.attend.set_layer_idx(layer_idx)
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states)
