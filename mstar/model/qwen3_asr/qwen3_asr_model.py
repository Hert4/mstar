"""Qwen3-ASR on M*: audio encoder then a dense Qwen3 decoder.

Graph (two nodes, one partition):

    prefill:  audio_encoder ──audio_embeds──► decoder ──new_token──► client
    decode:   Loop( decoder ──new_token──► client, ──text_inputs──► itself )

The decoder is an ordinary causal LM. The audio reaches it as tokens: the
prompt carries one ``<|audio_pad|>`` per encoder output frame and the
decoder overwrites those embeddings with the encoder's. So there is one
KV cache and nothing reserved for a window the clip does not fill —
which is the difference from the Whisper port, where cross-attention over
a fixed 1500-frame window costs 246 MB per request regardless of the
clip's real length. Here a 10 s clip is ~130 audio tokens, ~15 MB.

Prompt, from the checkpoint's chat template:

    <|im_start|>system\\n{context}<|im_end|>\\n
    <|im_start|>user\\n<|audio_start|><|audio_pad|>xN<|audio_end|><|im_end|>\\n
    <|im_start|>assistant\\n
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from mstar.communication.tensors import NameToTensorList, TensorPointerInfo
from mstar.conductor.request_info import (
    CurrentForwardConductorMetadata,
    StreamingConnectionState,
)
from mstar.engine.resources import (
    AttentionConfig,
    AttentionSpec,
    KVConfig,
    KVSpec,
    NodeResourceSpec,
    PositionConfig,
    PositionSpec,
    ResourceReqConfig,
    SamplerSpec,
    SamplingReqConfig,
)
from mstar.graph.base import GraphEdge, GraphNode, GraphSection, Loop, Sequential
from mstar.graph.special_destinations import EMIT_TO_CLIENT
from mstar.model.base import ForwardPassArgs, Model
from mstar.model.qwen3_asr.config import (
    ATTN,
    KV_CACHE,
    POS,
    SAMPLER,
    Qwen3ASRModelConfig,
)
from mstar.model.submodule_base import NodeSubmodule

logger = logging.getLogger(__name__)

# The model opens its answer with ``language {Name}<asr_text>``; forcing
# that prefix needs the full name the checkpoint was trained on, keyed by
# the ISO-639-1 code an OpenAI-shaped transcription request sends. Names
# are the checkpoint's own ``support_languages`` list.
ASR_TEXT_TAG = "<asr_text>"
SUPPORTED_LANGUAGES = {
    "zh": "Chinese", "en": "English", "yue": "Cantonese", "ar": "Arabic",
    "de": "German", "fr": "French", "es": "Spanish", "pt": "Portuguese",
    "id": "Indonesian", "it": "Italian", "ko": "Korean", "ru": "Russian",
    "th": "Thai", "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish",
    "hi": "Hindi", "ms": "Malay", "nl": "Dutch", "sv": "Swedish",
    "da": "Danish", "fi": "Finnish", "pl": "Polish", "cs": "Czech",
    "fil": "Filipino", "fa": "Persian", "el": "Greek", "ro": "Romanian",
    "hu": "Hungarian", "mk": "Macedonian",
}

# Longest sequence one request can reach: ~390 audio tokens for a full 30 s
# clip, ~20 of prompt scaffolding, and the transcript. 1024 covers that with
# room; it bounds pages per request, it does not reserve them.
MAX_SEQ_LEN = 1024


class Qwen3ASRModel(Model):
    def __init__(
        self,
        model_path_hf: str = "Qwen/Qwen3-ASR-0.6B",
        cache_dir: str | None = None,
        model_dir: str | None = None,
        **kwargs,
    ):
        """``model_dir`` is a local checkpoint directory, from the config
        YAML's ``model_kwargs``. It wins over ``model_path_hf`` so an
        air-gapped deployment can point at a staged copy without the Hub."""
        super().__init__()
        self.model_path_hf = model_path_hf
        self.cache_dir = cache_dir
        self.local_dir = self._resolve_dir(model_dir or model_path_hf, cache_dir)
        self.config = Qwen3ASRModelConfig.from_pretrained(self.local_dir)

        from transformers import AutoFeatureExtractor, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.local_dir)
        # The checkpoint's preprocessor_config declares WhisperFeatureExtractor
        # — same 128-bin, 30 s, hop-160 front end as Whisper.
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.local_dir)

        from mstar.model.utils import ByteLevelDetokenizer

        self._detokenizer = ByteLevelDetokenizer(self.tokenizer)
        self._submodule_cache: dict[str, NodeSubmodule] = {}
        self._prompt_cache: dict[tuple[str, int], list[int]] = {}

    @staticmethod
    def _resolve_dir(model_path: str, cache_dir: str | None) -> str:
        if Path(model_path).exists():
            return model_path
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=model_path, cache_dir=cache_dir)

    # -------------------------------------------------------------------
    # Model ABC: resources
    # -------------------------------------------------------------------

    def get_node_resources(self) -> list[NodeResourceSpec]:
        """One KV cache. The audio is in it, so it is sized by real audio.

        A page here is ``page_size`` tokens across 28 layers of 8 GQA KV
        heads — ~14 MB — and a 10 s request touches two of them. The
        Whisper port needs a second, larger cache for the encoder window;
        this model has no cross-attention and therefore no second stream.
        """
        kv_config = KVConfig(
            num_layers=self.config.num_hidden_layers,
            num_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.head_dim,
            max_seq_len=MAX_SEQ_LEN,
            num_qo_heads=self.config.num_attention_heads,
            max_num_pages=256,
        )
        return [
            KVSpec(resource_key=KV_CACHE, nodes={"decoder"}, config=kv_config),
            AttentionSpec(
                resource_key=ATTN, nodes={"decoder"},
                config=AttentionConfig(kv_cache=KV_CACHE),
            ),
            PositionSpec(
                resource_key=POS, nodes={"decoder"},
                # Plain 1-D RoPE: the checkpoint's mrope_section collapses to
                # this for audio+text. See config.py.
                config=PositionConfig(
                    kv_cache=KV_CACHE, rope_theta=self.config.rope_theta,
                ),
            ),
            SamplerSpec(
                resource_key=SAMPLER, nodes={"decoder"},
                vocab_size=self.config.vocab_size,
                enable_repetion_penalty=False,
            ),
        ]

    def get_request_resource_configs(
        self, partition_fwd_args: dict[str, ForwardPassArgs],
        model_kwargs: dict | None = None,
    ) -> dict[str, ResourceReqConfig]:
        del partition_fwd_args
        model_kwargs = model_kwargs or {}
        return {
            SAMPLER: SamplingReqConfig(
                # The checkpoint's generation_config is greedy.
                temperature=model_kwargs.get("temperature", 0.0),
                top_p=model_kwargs.get("top_p", 1.0),
                ignore_eos=model_kwargs.get("ignore_eos", False),
            )
        }

    # -------------------------------------------------------------------
    # Model ABC: graph
    # -------------------------------------------------------------------

    def get_max_output_tokens(self, **model_kwargs):
        # Leave room for the prompt inside MAX_SEQ_LEN; a 30 s clip's
        # ~390 audio tokens plus scaffolding is the worst case.
        limit = MAX_SEQ_LEN - self.config.audio_output_len(3000) - 32
        return min(model_kwargs.get("max_output_tokens", limit), limit)

    def get_graph_walk_graphs(self) -> dict[str, GraphSection]:
        prefill = Sequential([
            GraphNode(
                name="audio_encoder",
                input_names=["audio_features"],
                outputs=[GraphEdge(next_node="decoder", name="audio_embeds")],
            ),
            GraphNode(
                name="decoder",
                input_names=["audio_embeds", "text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                        persist=True,
                    ),
                ],
            ),
        ])

        decode = Loop(
            name="decode_loop",
            section=GraphNode(
                name="decoder",
                input_names=["text_inputs"],
                outputs=[
                    GraphEdge(
                        next_node=EMIT_TO_CLIENT,
                        name="new_token",
                        output_modality="text",
                    ),
                    GraphEdge(next_node="decoder", name="text_inputs"),
                ],
            ),
            max_iters=self.get_max_output_tokens(),
            outputs=[],
        )

        return dict(prefill=prefill, decode=decode)

    # -------------------------------------------------------------------
    # Model ABC: forward pass args
    # -------------------------------------------------------------------

    def get_initial_forward_pass_args(
        self,
        partition_name: str,
        input_modalities: list[str],
        output_modalities: list[str],
        input_signals: dict[str, list[TensorPointerInfo]],
        model_kwargs: dict | None = None,
    ) -> ForwardPassArgs:
        full_metadata = CurrentForwardConductorMetadata(
            input_modalities=input_modalities,
            output_modalities=output_modalities,
            graph_walk="prefill",
            is_prefill=True,
        )

        audio_edge = GraphEdge(next_node="audio_encoder", name="audio_features")
        audio_edge.tensor_info = input_signals.get("audio_features", [])
        text_edge = GraphEdge(next_node="decoder", name="text_inputs")
        text_edge.tensor_info = input_signals.get("text_inputs", [])
        inputs = [audio_edge, text_edge]

        return ForwardPassArgs(
            full_metadata=full_metadata,
            inputs=inputs,
            unpersist_tensors=sum([i.tensor_info for i in inputs], start=[]),
            step_metadata={"is_prefill": True},
        )

    def get_partition_forward_pass_args(
        self,
        partition_name: str,
        partition_metadata: CurrentForwardConductorMetadata,
        persist_signals: dict[str, list[TensorPointerInfo]],
        incoming_connections: list[StreamingConnectionState] | None = None,
    ) -> ForwardPassArgs:
        """Single-partition state machine: prefill -> decode loop -> done."""
        metadata = partition_metadata

        if metadata.is_prefill:
            metadata.is_prefill = False
            metadata.graph_walk = "decode"
        elif metadata.graph_walk == "decode":
            return ForwardPassArgs(
                full_metadata=metadata,
                inputs=[],
                unpersist_tensors=[],
                request_done=True,
            )

        edge = GraphEdge(next_node="decoder", name="text_inputs")
        edge.tensor_info = persist_signals.get("new_token", [])
        inputs = [edge]

        return ForwardPassArgs(
            full_metadata=metadata,
            inputs=inputs,
            unpersist_tensors=sum([i.tensor_info for i in inputs], start=[]),
            step_metadata={"is_prefill": False},
        )

    # -------------------------------------------------------------------
    # Model ABC: prompt processing
    # -------------------------------------------------------------------

    def _build_prompt_ids(
        self, num_audio_tokens: int, context: str, language: str | None,
    ) -> list[int]:
        """Chat-template prompt with one audio placeholder per encoder frame.

        Built from ids rather than by rendering the Jinja template: the
        template emits a single ``<|audio_pad|>`` that the HF processor
        later expands, and doing the expansion here keeps the count and
        the encoder's output length derived from the same number.

        Two details are taken from the reference rather than the template,
        because the template does not express them and both were measured
        to matter (vllm ``qwen3_asr.get_generation_prompt``):

        * the system turn is omitted entirely when there is no context,
          not emitted empty;
        * a known ``language`` prefills the assistant turn with
          ``language {Name}<asr_text>``. The model otherwise opens by
          guessing the language itself, which is both several tokens of
          generation and a decision that can go the wrong way — measured
          5.1% WER without it against 3.1% with, on the same checkpoint.
        """
        key = (context, num_audio_tokens, language or "")
        cached = self._prompt_cache.get(key)
        if cached is not None:
            return cached

        enc = self.tokenizer.encode
        ids: list[int] = []
        if context:
            ids += enc(f"<|im_start|>system\n{context}<|im_end|>\n",
                       add_special_tokens=False)
        ids += enc("<|im_start|>user\n", add_special_tokens=False)
        ids += [self.config.audio_start_token_id]
        ids += [self.config.audio_token_id] * num_audio_tokens
        ids += [self.config.audio_end_token_id]
        ids += enc("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
        if language:
            name = SUPPORTED_LANGUAGES.get(language.lower(), language)
            ids += enc(f"language {name}{ASR_TEXT_TAG}", add_special_tokens=False)

        if len(self._prompt_cache) < 256:
            self._prompt_cache[key] = ids
        return ids

    def process_prompt(
        self,
        prompt: str | None,
        input_modalities: list[str],
        output_modalities: list[str],
        tensors: NameToTensorList | None = None,
        **kwargs,
    ) -> NameToTensorList:
        raw_audio_inputs = (tensors or {}).get("audio_inputs", [])
        if len(raw_audio_inputs) != 1:
            raise ValueError(
                f"Qwen3-ASR expects exactly one audio input per request; "
                f"got {len(raw_audio_inputs)}."
            )

        feat = self.feature_extractor(
            raw_audio_inputs[0].cpu().numpy(),
            sampling_rate=self.feature_extractor.sampling_rate,
            return_attention_mask=True,
            padding="max_length",
            return_tensors="pt",
        )
        # The extractor pads to the fixed 30 s window; trim back to the real
        # frames. Keeping the padding would put the model back where Whisper
        # is — paying for 30 s of audio that is not there — and would also
        # make the placeholder count wrong.
        mask = feat.get("attention_mask")
        frames = int(mask[0].sum()) if mask is not None else feat["input_features"].shape[-1]
        audio_features = feat["input_features"][0][:, :frames]

        num_audio_tokens = self.config.audio_output_len(frames)
        prompt_ids = self._build_prompt_ids(
            num_audio_tokens,
            context=kwargs.get("context", "") or "",
            language=kwargs.get("language") or None,
        )

        return {
            "audio_features": [audio_features],
            "text_inputs": [torch.tensor(prompt_ids, dtype=torch.long)],
        }

    def postprocess(self, output: torch.Tensor, modality: str, **kwargs) -> bytes:
        if modality == "text":
            return self._detokenizer.to_bytes(output.reshape(-1).tolist())
        raise ValueError(f"Unsupported modality for Qwen3-ASR: {modality!r}")

    # -------------------------------------------------------------------
    # Model ABC: submodule loading
    # -------------------------------------------------------------------

    def get_submodule(
        self, node_name: str, device: str = "cpu", tp_group=None,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name in self._submodule_cache:
            return self._submodule_cache[node_name]
        submodule = self._create_submodule(
            node_name, device, autocast_dtype=autocast_dtype,
        )
        if submodule is not None:
            logger.info("Loaded Qwen3-ASR submodule for %s", node_name)
            self._submodule_cache[node_name] = submodule
        return submodule

    def _create_submodule(
        self, node_name: str, device: str,
        autocast_dtype: torch.dtype | None = None,
    ) -> NodeSubmodule | None:
        if node_name == "audio_encoder":
            return self._create_encoder_submodule(device, autocast_dtype)
        if node_name == "decoder":
            return self._create_decoder_submodule(device, autocast_dtype)
        return None

    def _create_encoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None,
    ) -> NodeSubmodule:
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoderConfig,
        )
        from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
            Qwen3OmniMoeAudioEncoder,
        )

        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_asr.submodules import Qwen3ASRAudioEncoderSubmodule

        # Qwen3-ASR reuses Qwen3-Omni's audio tower verbatim; the reference
        # implementation instantiates the same class from this same config
        # block (vllm qwen3_asr: ``self.audio_tower = Qwen3OmniMoeAudioEncoder``).
        hf_config = Qwen3OmniMoeAudioEncoderConfig(**self.config.audio_config)
        with torch.device("meta"):
            encoder = Qwen3OmniMoeAudioEncoder._from_config(
                hf_config, attn_implementation="sdpa",
            )
        if autocast_dtype is not None:
            encoder = encoder.to(autocast_dtype)
        encoder.to_empty(device=device)

        weights = iter_safetensors_shards(
            self.local_dir, device=device, prefix="thinker.audio_tower.",
        )
        weights = ((k.removeprefix("thinker.audio_tower."), v) for k, v in weights)
        load_hf_weights(encoder, weights)
        encoder.eval()

        return Qwen3ASRAudioEncoderSubmodule(
            audio_encoder=encoder, config=self.config,
        )

    def _create_decoder_submodule(
        self, device: str, autocast_dtype: torch.dtype | None,
    ) -> NodeSubmodule:
        from mstar.model.loader import load_hf_weights
        from mstar.model.loader.iterators import iter_safetensors_shards
        from mstar.model.qwen3_asr.components.decoder import Qwen3ASRDecoder
        from mstar.model.qwen3_asr.submodules import Qwen3ASRDecoderSubmodule

        with torch.device("meta"):
            decoder = Qwen3ASRDecoder(self.config)
        if autocast_dtype is not None:
            decoder = decoder.to(autocast_dtype)
        decoder.to_empty(device=device)

        def _renamed():
            for key, value in iter_safetensors_shards(
                self.local_dir, device=device, prefix="thinker.",
            ):
                if key == "thinker.lm_head.weight":
                    yield "lm_head_weight", value
                elif key.startswith("thinker.model."):
                    yield key.removeprefix("thinker.model."), value
                # thinker.audio_tower.* belongs to the other submodule

        load_hf_weights(decoder, _renamed())
        decoder.eval()

        return Qwen3ASRDecoderSubmodule(decoder=decoder, config=self.config)
