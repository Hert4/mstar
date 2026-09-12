"""WhisperEncoderSubmodule batches, and the batched path matches the single.

The encoder inherited ``can_batch = False`` and the base ``preprocess``, which
raises on more than one input, so it ran 1500 encoder positions once per
request. That cost never amortised, and since a request only reaches the
decoder after its own encoder pass, it also paced arrivals into the decode
loop — the decode batch could not fill either.

Batching is only sound because the window is fixed: ``process_prompt`` pads or
truncates every clip to 30 s, so a row is always ``(num_mel_bins, 3000)``.
These tests pin that, the per-request split, and parity with ``forward``.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

import torch
from torch import nn

from mstar.model.submodule_base import ModelInputsFromEngine, NodeInputs
from mstar.model.whisper.config import WhisperModelConfig
from mstar.model.whisper.submodules import WhisperEncoderSubmodule

MELS, FRAMES, POS, DMODEL = 4, 12, 5, 3


class _Encoder(nn.Module):
    """Stands in for the HF encoder: (B, mels, frames) -> (B, POS, DMODEL).

    Every output element carries the row's first mel value, so a split that
    hands a request another row's states is visible rather than merely
    differently shaped.
    """

    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))

    def forward(self, feats: torch.Tensor):
        tag = feats[:, :1, :1]  # (B, 1, 1)
        return SimpleNamespace(
            last_hidden_state=tag.expand(feats.shape[0], POS, DMODEL).clone()
        )


def _submodule() -> WhisperEncoderSubmodule:
    return WhisperEncoderSubmodule(_Encoder(), WhisperModelConfig())


def _rows(n: int) -> list[NodeInputs]:
    """Row i is filled with the value i, so it is traceable through the split."""
    return [
        NodeInputs(tensor_inputs={"audio_features": torch.full((MELS, FRAMES), float(i))})
        for i in range(n)
    ]


def _engine_inputs(n: int) -> ModelInputsFromEngine:
    rids = [f"r{i}" for i in range(n)]
    return ModelInputsFromEngine(request_ids=rids, per_request_info={})


def test_the_encoder_advertises_batching():
    """The defaults it used to inherit were False and None."""
    sub = _submodule()
    assert sub.can_batch(batch=None, model_inputs=_rows(4)) is True
    assert sub.max_batch_size("prefill") == WhisperEncoderSubmodule.MAX_BATCH_SIZE


def test_preprocess_stacks_the_ready_set():
    sub = _submodule()

    out = sub.preprocess("prefill", _engine_inputs(3), _rows(3))

    assert out["audio_features"].shape == (3, MELS, FRAMES)


def test_each_request_gets_its_own_encoder_states():
    """The split is by position in ``request_ids``; a transposed or shared
    slice would still have the right shape, so check the contents."""
    sub = _submodule()
    inputs = _rows(4)
    preprocessed = sub.preprocess("prefill", _engine_inputs(4), inputs)

    out = sub.forward_batched(
        "prefill", engine_inputs=_engine_inputs(4), **preprocessed
    )

    assert list(out) == ["r0", "r1", "r2", "r3"]
    for i, rid in enumerate(out):
        states = out[rid]["encoder_states"][0]
        assert states.shape == (POS, DMODEL)
        assert torch.equal(states, torch.full((POS, DMODEL), float(i)))


def test_the_batched_path_matches_the_single_request_one():
    """A batch of one must be indistinguishable from the unbatched forward,
    which is the path a lone request still takes."""
    sub = _submodule()
    row = _rows(1)

    single = sub.forward(
        "prefill", engine_inputs=_engine_inputs(1),
        **{"audio_features": row[0].tensor_inputs["audio_features"]},
    )["encoder_states"][0]
    batched = sub.forward_batched(
        "prefill", engine_inputs=_engine_inputs(1),
        **sub.preprocess("prefill", _engine_inputs(1), row),
    )["r0"]["encoder_states"][0]

    assert torch.equal(single, batched)
