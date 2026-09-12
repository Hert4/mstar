"""Config for Qwen3-ASR (``Qwen3ASRForConditionalGeneration``).

The checkpoint nests everything under ``thinker_config``: an
``audio_config`` for the encoder and a ``text_config`` for a plain dense
Qwen3 decoder. This class flattens the two halves the runtime needs and
leaves the encoder's own config to HF, which owns that module.

Why this model is shaped differently from Whisper, which is the other
ASR here: Whisper feeds audio to the decoder through cross-attention over
a fixed 1500-frame window, so every request reserves cross-K/V for the
full window whatever the clip's real length — 246 MB each, 32 layers
wide. Qwen3-ASR instead projects the encoder output into the text
embedding space and splices it in at ``audio_token_id`` positions, so
audio lives in the ordinary KV cache and costs what the audio actually
is: ~13 tokens per second of speech, ~112 KB per token. A 10 s clip is
~15 MB. There is no cross-attention resource at all.

MRoPE: ``text_config.rope_scaling`` declares ``mrope_section`` and
``mrope_interleaved``, but the reference implementation assigns all three
axes the same value and keeps positions contiguous across the text and
audio segments (vLLM ``qwen3_asr.get_mrope_input_positions``). For an
audio-plus-text model there is no second or third spatial axis to carry,
so it degenerates exactly to 1-D sequential RoPE and the runtime plans it
with the ordinary position resource.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Resource labels (decoder node)
# ---------------------------------------------------------------------------
KV_CACHE = "kv_cache"
ATTN = "attn"
POS = "rope"
SAMPLER = "sampler"


@dataclass
class Qwen3ASRModelConfig:
    # --- text decoder (thinker_config.text_config, model_type "qwen3") ---
    hidden_size: int = 1024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    intermediate_size: int = 3072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    vocab_size: int = 151936
    max_position_embeddings: int = 65536
    tie_word_embeddings: bool = True

    # --- audio encoder (thinker_config.audio_config) ---
    # Kept as a plain dict: it is handed to HF's
    # ``Qwen3OmniMoeAudioEncoderConfig``, which owns these fields.
    audio_config: dict = field(default_factory=dict)
    num_mel_bins: int = 128
    # Mel frames in the fixed feature-extractor window. Only used to size
    # the dummy input for warmup; the real length drives everything else.
    max_source_frames: int = 3000

    # --- multimodal splice points (thinker_config) ---
    audio_start_token_id: int = 151669
    audio_end_token_id: int = 151670
    audio_token_id: int = 151676

    # --- generation ---
    eos_token_ids: tuple[int, ...] = (151643, 151645)

    @property
    def kv_bytes_per_token(self) -> int:
        """One token's paged-KV footprint across the whole decoder."""
        return self.num_hidden_layers * 2 * self.num_key_value_heads * self.head_dim * 2

    @staticmethod
    def audio_output_len(num_mel_frames: int) -> int:
        """Encoder output tokens for ``num_mel_frames`` of *real* audio.

        Verbatim from the reference
        (``vllm/model_executor/models/qwen3_asr._get_feat_extract_output_lengths``):
        the encoder's conv stack strides the tail, and whole 100-frame
        blocks collapse to 13 each. Roughly 13 tokens per second.

        The count has to match the number of ``audio_token_id``
        placeholders the prompt carries, or the splice misaligns — hence
        the same arithmetic rather than a derived approximation.
        """
        leave = num_mel_frames % 100
        feat = (leave - 1) // 2 + 1
        return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (num_mel_frames // 100) * 13

    @classmethod
    def from_pretrained(cls, local_dir: str | Path) -> "Qwen3ASRModelConfig":
        with open(Path(local_dir) / "config.json") as f:
            hf = json.load(f)
        thinker = hf.get("thinker_config", hf)
        text = thinker.get("text_config", {})
        audio = thinker.get("audio_config", {})

        gen: dict = {}
        gen_path = Path(local_dir) / "generation_config.json"
        if gen_path.exists():
            with open(gen_path) as f:
                gen = json.load(f)
        eos = gen.get("eos_token_id", list(cls.eos_token_ids))
        if isinstance(eos, int):
            eos = [eos]

        return cls(
            hidden_size=text["hidden_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            # Qwen3 carries head_dim explicitly; it is not hidden/heads.
            head_dim=text.get("head_dim", text["hidden_size"] // text["num_attention_heads"]),
            intermediate_size=text["intermediate_size"],
            rms_norm_eps=text.get("rms_norm_eps", 1e-6),
            rope_theta=float(text.get("rope_theta", 1e6)),
            vocab_size=text["vocab_size"],
            max_position_embeddings=text.get("max_position_embeddings", 65536),
            tie_word_embeddings=text.get("tie_word_embeddings", True),
            audio_config=audio,
            num_mel_bins=audio.get("num_mel_bins", 128),
            audio_start_token_id=thinker["audio_start_token_id"],
            audio_end_token_id=thinker["audio_end_token_id"],
            audio_token_id=thinker["audio_token_id"],
            eos_token_ids=tuple(eos),
        )
