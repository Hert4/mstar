"""Qwen3-ASR: audio token accounting and the embedding splice.

The prompt carries one ``<|audio_pad|>`` per encoder output frame, and
the decoder overwrites exactly those positions with the encoder's
output. Nothing checks that alignment at runtime except the splice
itself, and a silent off-by-one there would shift every later token
rather than fail — so the count is pinned here from both ends.

``audio_output_len`` is the reference arithmetic
(vllm ``qwen3_asr._get_feat_extract_output_lengths``); these cases are
the ones where its two branches disagree — a whole number of 100-frame
blocks, and a remainder that the conv stack strides down separately.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import pytest
import torch

from mstar.model.qwen3_asr.components.decoder import Qwen3ASRDecoder
from mstar.model.qwen3_asr.config import Qwen3ASRModelConfig


def _config(**over) -> Qwen3ASRModelConfig:
    """A miniature of the real layout: same field meanings, small enough
    to instantiate on meta without the 0.6B checkpoint."""
    base = dict(
        hidden_size=8, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, intermediate_size=16,
        vocab_size=64, audio_token_id=60,
    )
    base.update(over)
    return Qwen3ASRModelConfig(**base)


def _reference_output_len(n: int) -> int:
    """vllm/model_executor/models/qwen3_asr._get_feat_extract_output_lengths."""
    leave = n % 100
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (n // 100) * 13


@pytest.mark.parametrize("frames", [0, 1, 50, 99, 100, 101, 300, 999, 1000, 1500, 3000])
def test_audio_output_len_tracks_the_reference(frames):
    assert Qwen3ASRModelConfig.audio_output_len(frames) == _reference_output_len(frames)


def test_a_second_of_audio_is_about_thirteen_tokens():
    """Sanity on the scale, which is the whole reason this model is cheaper
    than Whisper: 100 mel frames is one second at hop 160 / 16 kHz."""
    assert Qwen3ASRModelConfig.audio_output_len(1000) == 130
    assert Qwen3ASRModelConfig.audio_output_len(3000) == 390


def test_kv_per_token_is_the_real_layout():
    """28 layers x K,V x 8 GQA heads x 128 dim x 2 bytes = 112 KiB."""
    assert Qwen3ASRModelConfig().kv_bytes_per_token == 28 * 2 * 8 * 128 * 2


def _decoder(config):
    with torch.device("meta"):
        decoder = Qwen3ASRDecoder(config)
    return decoder.to_empty(device="cpu")


def test_the_splice_lands_on_the_placeholders_only():
    config = _config()
    decoder = _decoder(config)
    input_ids = torch.tensor([1, 2, config.audio_token_id, config.audio_token_id, 3])
    embeds = torch.zeros(5, config.hidden_size)
    audio = torch.arange(1, 3, dtype=torch.float32).repeat_interleave(
        config.hidden_size
    ).reshape(2, config.hidden_size)

    out = decoder.splice_audio(embeds, input_ids, audio)

    assert torch.equal(out[2], audio[0]), "first placeholder takes the first row"
    assert torch.equal(out[3], audio[1]), "and the second takes the second"
    assert not out[[0, 1, 4]].any(), "text positions are untouched"
    assert not embeds.any(), "and the caller's tensor is not mutated"


def test_a_count_mismatch_raises_rather_than_shifting_the_sequence():
    """The failure this guards is silent: too few rows would leave a
    placeholder embedding in place and every later token would attend to
    it as if it were audio."""
    config = _config()
    decoder = _decoder(config)
    input_ids = torch.tensor([config.audio_token_id] * 3)
    embeds = torch.zeros(3, config.hidden_size)

    with pytest.raises(ValueError, match="3 audio placeholders"):
        decoder.splice_audio(embeds, input_ids, torch.zeros(2, config.hidden_size))


def test_a_prompt_with_no_audio_is_left_alone():
    config = _config()
    decoder = _decoder(config)
    input_ids = torch.tensor([1, 2, 3])
    embeds = torch.randn(3, config.hidden_size)

    out = decoder.splice_audio(embeds, input_ids, torch.zeros(0, config.hidden_size))

    assert torch.equal(out, embeds)


def test_from_pretrained_reads_the_nested_thinker_layout(tmp_path):
    """The checkpoint nests the two halves under ``thinker_config``; a flat
    read would silently take defaults for all of them."""
    import json

    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "qwen3_asr",
        "thinker_config": {
            "audio_start_token_id": 11, "audio_end_token_id": 12,
            "audio_token_id": 13,
            "audio_config": {"num_mel_bins": 128, "d_model": 896},
            "text_config": {
                "hidden_size": 1024, "num_hidden_layers": 28,
                "num_attention_heads": 16, "num_key_value_heads": 8,
                "head_dim": 128, "intermediate_size": 3072,
                "vocab_size": 151936, "rope_theta": 1000000,
            },
        },
    }))
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [151643, 151645]})
    )

    config = Qwen3ASRModelConfig.from_pretrained(tmp_path)

    assert config.head_dim == 128, "Qwen3 states head_dim; it is not hidden/heads"
    assert config.num_key_value_heads == 8
    assert config.rope_theta == 1_000_000.0
    assert config.audio_token_id == 13
    assert config.eos_token_ids == (151643, 151645)
    assert config.audio_config["d_model"] == 896
