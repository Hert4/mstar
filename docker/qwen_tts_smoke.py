"""Build-time smoke test for the Qwen3-TTS 12 Hz codec.

Runs a real forward, not just a construction. The adaptations this image
applies to the qwen-tts sources are for a transformers major-version gap,
and that gap does not show itself all at once: the import-time breakages
(decorator, rope registry) surfaced first, and a fourth one lived inside
``forward`` and only appeared on a served request — after a build, a
registry mirror and a deploy. This test moves that discovery to the build.

The config is the decoder_config from the published checkpoint's
speech_tokenizer/config.json, inlined because the checkpoint is not in
the image. It only has to be structurally faithful: weights are random,
the point is that every call site the forward touches still exists.
"""
import torch

from mstar.model.qwen3_tts.qwen3_tts_model import _load_qwen3_tts_decoder_classes

DECODER_CONFIG = {
    "attention_bias": False, "attention_dropout": 0.0, "latent_dim": 1024,
    "codebook_dim": 512, "codebook_size": 2048, "decoder_dim": 1536,
    "hidden_act": "silu", "hidden_size": 512, "intermediate_size": 1024,
    "layer_scale_initial_scale": 0.01, "max_position_embeddings": 8000,
    "head_dim": 64, "num_attention_heads": 16, "num_hidden_layers": 8,
    "num_key_value_heads": 16, "num_quantizers": 16,
    "num_semantic_quantizers": 1, "rms_norm_eps": 1e-05, "rope_theta": 10000,
    "semantic_codebook_size": 4096, "sliding_window": 72,
    "upsample_rates": [8, 5, 4, 3], "upsampling_ratios": [2, 2],
    "vector_quantization_hidden_dimension": 512,
}

config_cls, decoder_cls = _load_qwen3_tts_decoder_classes()
config = config_cls(**DECODER_CONFIG)
decoder = decoder_cls(config).eval()

frames = 8
codes = torch.randint(0, config.codebook_size, (1, DECODER_CONFIG["num_quantizers"], frames))
with torch.no_grad():
    out = decoder(codes)
audio = out[0] if isinstance(out, (tuple, list)) else getattr(out, "audio_values", out)

expected = frames * 1920  # decode_upsample_rate
assert torch.is_tensor(audio), type(audio)
assert audio.shape[-1] == expected, (audio.shape, expected)
print(f"qwen-tts 12hz codec forward ok: {tuple(audio.shape)}, "
      f"{sum(p.numel() for p in decoder.parameters()) / 1e6:.0f}M params")
