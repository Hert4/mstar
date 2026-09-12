"""Qwen3-ASR as two graph nodes: audio encoder, then an AR text decoder.

Same shape as the Whisper port next door, with the cross-attention half
removed. The encoder's output is projected into the text embedding space
and written over the ``audio_token_id`` placeholders the prompt carries,
so the decoder is an ordinary causal LM with one KV cache.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from mstar.communication.tensors import NameToTensorList
from mstar.conductor.request_info import CurrentForwardPassInfo
from mstar.engine.cuda_graph_config import BatchedCudaGraphConfig, CudaGraphConfig
from mstar.engine.engine import ExecutingBatch
from mstar.engine.resources import (
    AttentionStep,
    KVStep,
    PositionStep,
    SamplerStep,
    Segment,
    SlotLease,
    SubmoduleStep,
)
from mstar.engine.resources.attn.base import AttentionManager
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.qwen3_asr.components.decoder import Qwen3ASRDecoder
from mstar.model.qwen3_asr.config import (
    ATTN,
    KV_CACHE,
    POS,
    SAMPLER,
    Qwen3ASRModelConfig,
)
from mstar.model.submodule_base import (
    ARNodeInputs,
    ARNodeSubmodule,
    ModelInputsFromEngine,
    NodeInputs,
    NodeSubmodule,
)

logger = logging.getLogger(__name__)


class Qwen3ASRAudioEncoderSubmodule(NodeSubmodule):
    """HF ``Qwen3OmniMoeAudioEncoder``, run once per request at prefill.

    The HF module already takes a packed, variable-length batch
    (``input_features`` concatenated along time plus ``feature_lens``)
    and returns the outputs packed the same way, so batching here is a
    concatenate and a split rather than a pad.

    That is worth saying because the Whisper encoder in this tree is the
    opposite: it takes a fixed ``(B, mel, 3000)`` block, so every clip
    pays for a full 30 s window whether or not it is one.
    """

    # An encoder pass allocates fresh activations proportional to the batch,
    # unlike a decode step reading a KV cache that was already reserved — so
    # this cap is a memory budget, not a throughput knob, and it must not be
    # copied from the decoder's. Conservative here because the packed
    # interface means a "batch of 4" can still be 4 x 30 s of audio.
    MAX_BATCH_SIZE = 4

    # forward_batched returns a dict keyed by request id, and those keys
    # change every call, so dynamo re-specialises on each one and never
    # reaches a steady compiled state. See the Whisper encoder for the
    # measurement.
    disable_torch_compile = True

    def __init__(self, audio_encoder: nn.Module, config: Qwen3ASRModelConfig):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.config = config

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> NodeInputs:
        return NodeInputs(
            tensor_inputs={"audio_features": inputs["audio_features"][0]}
        )

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    @staticmethod
    def _frames(feat: torch.Tensor) -> int:
        """Mel frames in one request's features, laid out (mel_bins, frames)."""
        return feat.shape[-1]

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[NodeInputs],
    ) -> dict:
        del graph_walk, engine_inputs
        feats = [inp.tensor_inputs["audio_features"] for inp in inputs]
        return {
            # (mel_bins, sum_frames): the packed layout the HF module wants
            "audio_features": torch.cat(feats, dim=-1),
            "feature_lens": torch.tensor(
                [self._frames(f) for f in feats], dtype=torch.long,
            ),
        }

    def _encode(
        self, audio_features: torch.Tensor, feature_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, list[int]]:
        device = self.get_device()
        dtype = next(self.audio_encoder.parameters()).dtype
        feats = audio_features.to(device=device, dtype=dtype)
        lens = feature_lens.to(device=device)
        out_lens = [
            self.config.audio_output_len(int(n)) for n in feature_lens.tolist()
        ]
        states = self.audio_encoder(
            feats,
            feature_lens=lens,
            aftercnn_lens=torch.tensor(out_lens, dtype=torch.long, device=device),
        )
        # HF returns either a bare tensor or a wrapper with last_hidden_state,
        # depending on the transformers minor; both appear across the images
        # this runs on.
        if not torch.is_tensor(states):
            states = states.last_hidden_state
        return states, out_lens

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        feature_lens: torch.Tensor,
        **kwargs,
    ) -> NameToTensorList:
        del graph_walk, engine_inputs
        states, _ = self._encode(audio_features, feature_lens)
        return {"audio_embeds": [states]}

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        audio_features: torch.Tensor,
        feature_lens: torch.Tensor,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        del graph_walk
        states, out_lens = self._encode(audio_features, feature_lens)
        parts = states.split(out_lens)
        return {
            rid: {"audio_embeds": [parts[i]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }


class Qwen3ASRDecoderSubmodule(ARNodeSubmodule):
    """Dense Qwen3 decoder over one paged KV cache.

    prefill: embed the prompt, overwrite the ``audio_token_id`` slots with
    the encoder output, run the stack, sample the first token.
    decode: embed the previous token, single step.

    There is no second cache stream. Whisper needs one because its audio
    reaches the decoder through cross-attention over a fixed window;
    here the audio is already tokens, so it lives in ``main`` with
    everything else and costs what its real length costs.
    """

    DECODE_CAPTURE_BATCH_SIZES = [1, 2, 4, 8, 16]

    def __init__(self, decoder: Qwen3ASRDecoder, config: Qwen3ASRModelConfig):
        super().__init__()
        self.decoder = decoder
        self.config = config
        self._eos = torch.tensor(sorted(config.eos_token_ids), dtype=torch.long)

    def get_cuda_graph_configs(
        self, device: torch.device, tp_world_size: int = 1,
    ) -> list[CudaGraphConfig]:
        return [
            BatchedCudaGraphConfig(
                capture_graph_walk="decode",
                single_request_inputs=ARNodeInputs(
                    input_ids=torch.zeros(1, dtype=torch.long, device=device),
                    input_seq_len=1,
                ),
                capture_batch_sizes=self.DECODE_CAPTURE_BATCH_SIZES,
            ),
        ]

    def prepare_inputs(
        self,
        graph_walk: str,
        fwd_info: CurrentForwardPassInfo,
        inputs: NameToTensorList,
        **kwargs,
    ) -> ARNodeInputs:
        device = self.get_device()
        token_ids = inputs["text_inputs"][0].to(device).reshape(-1)

        tensor_inputs = {}
        if graph_walk == "prefill":
            tensor_inputs["audio_embeds"] = inputs["audio_embeds"][0].to(device)

        return ARNodeInputs(
            input_ids=token_ids,
            input_seq_len=token_ids.shape[0],
            tensor_inputs=tensor_inputs,
        )

    def declare_step(
        self,
        graph_walk: str,
        request_ids: list[str],
        inputs: list[ARNodeInputs],
        slot_lease: SlotLease | None = None,
        piecewise_leases: Mapping[str, SlotLease] | None = None,
        **kwargs,
    ) -> SubmoduleStep:
        return SubmoduleStep(
            segments=[
                Segment(request_id=rid, label="main", span=inp.input_seq_len)
                for rid, inp in zip(request_ids, inputs, strict=True)
            ],
            steps={
                KV_CACHE: KVStep(),
                ATTN: AttentionStep(causal=True),
                SAMPLER: SamplerStep(apply_penalty=False),
                POS: PositionStep(),
            },
        )

    def preprocess(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        inputs: list[ARNodeInputs],
    ) -> dict[str, torch.Tensor | Any]:
        preprocessed: dict[str, torch.Tensor | Any] = {
            "input_ids": torch.cat([inp.input_ids for inp in inputs]),
        }
        if graph_walk == "prefill":
            # Concatenated in the same request order as input_ids, so the
            # placeholder mask over the packed sequence lines up with these
            # rows without carrying per-request offsets.
            preprocessed["audio_embeds"] = torch.cat(
                [inp.tensor_inputs["audio_embeds"] for inp in inputs], dim=0,
            )
        return preprocessed

    def _forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        attn: AttentionManager = engine_inputs.resources[ATTN]
        sampler: SamplerResource = engine_inputs.resources[SAMPLER]

        input_embeds = self.decoder.embed(input_ids)
        if audio_embeds is not None:
            input_embeds = self.decoder.splice_audio(
                input_embeds, input_ids, audio_embeds,
            )

        hidden = self.decoder(input_embeds=input_embeds, label="main")
        if graph_walk == "prefill":
            # packed prefill: one hidden per request, at its last token
            hidden = attn.select_last_hidden(hidden)

        logits = self.decoder.lm_head(hidden)
        return sampler.sample(engine_inputs.request_ids, logits=logits)

    def forward(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> NameToTensorList:
        return {
            "new_token": [self._forward(
                graph_walk=graph_walk,
                engine_inputs=engine_inputs,
                input_ids=input_ids,
                audio_embeds=audio_embeds,
            )]
        }

    # Prefill is not captured, so the capture cap does not bound it; this is
    # what keeps a ready set from exceeding the context cache.
    MAX_BATCH_SIZE = 16

    def can_batch(
        self, batch: ExecutingBatch, model_inputs: list[NodeInputs],
    ) -> bool:
        return True

    def max_batch_size(self, graph_walk: str) -> int | None:
        del graph_walk
        return self.MAX_BATCH_SIZE

    def graph_walk_priority(self, graph_walk: str) -> int:
        """Decode ahead of prefill; see ``NodeSubmodule.graph_walk_priority``.

        Measured on the Whisper port, a prefill step costs roughly an order
        of magnitude more than a decode step, and round-robin over equal
        turns hands prefill most of the device. The same shape applies here.
        """
        return 1 if graph_walk == "decode" else 0

    def forward_batched(
        self,
        graph_walk: str,
        engine_inputs: ModelInputsFromEngine,
        input_ids: torch.Tensor,
        audio_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, NameToTensorList]:
        new_tokens = self._forward(
            graph_walk=graph_walk,
            engine_inputs=engine_inputs,
            input_ids=input_ids,
            audio_embeds=audio_embeds,
        )
        return {
            rid: {"new_token": [new_tokens[i:i + 1]]}
            for i, rid in enumerate(engine_inputs.request_ids)
        }

    def postprocess(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
        **kwargs,
    ):
        # Metadata-only: rebind the sampled token so the decode loop feeds it
        # back in as the next step's text_inputs.
        if "new_token" not in outputs:
            return
        outputs["text_inputs"] = outputs["new_token"]

    def check_stop(
        self,
        request_id: str,
        request_info: CurrentForwardPassInfo,
        outputs: dict[str, list[torch.Tensor]],
    ) -> set[str]:
        if "new_token" not in outputs:
            return set()
        # Reads a CPU copy the worker pre-materialised on a side stream, so
        # this does not sync the default stream.
        token = outputs["new_token"][0].item()
        ignore_eos = request_info.resource_configs[SAMPLER].ignore_eos
        decoded = request_info.dynamic_loop_iter_counts.get("decode_loop", 0) + 1
        # Qwen3-ASR ends on either of two ids (<|endoftext|>, <|im_end|>).
        if (not ignore_eos and token in self.config.eos_token_ids) or \
                decoded >= request_info.max_tokens:
            return {"decode_loop"}
        return set()
