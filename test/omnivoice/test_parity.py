"""Parity of the OmniVoice port against the reference implementation.

Three tiers, so the cheap checks still run where there is no GPU:

``test_canvas_batch_has_no_fully_masked_row``
    Pure tensor shapes, no weights, no network.  Guards the padding hazard: a
    query row with no visible key softmaxes to NaN, and since a masked key
    still contributes ``0 * NaN`` to every other query, one such row poisons
    the whole document at the next layer.

``test_prefix_parity``
    Tokenizer only.  The canvas prefix must be token-identical to the
    reference's ``_prepare_inference_inputs``, since a drift there changes
    everything downstream.

``test_unmask_parity``
    Real weights on a GPU.  Runs both paths fully greedy — ``position_temperature``
    and ``class_temperature`` both zero, which removes every source of
    randomness — and demands a token-exact canvas.

Run the first two anywhere::

    pytest test/omnivoice/test_parity.py -k "not unmask"

and the full set on a box with the checkpoint::

    OMNIVOICE_PATH=/path/to/OmniVoice pytest test/omnivoice/test_parity.py
"""

import os

import pytest
import torch

from mstar.model.omnivoice.components.backbone import CanvasItem, build_canvas_batch
from mstar.model.omnivoice.components.text import build_prefix
from mstar.model.omnivoice.components.unmask import (
    build_reveal_schedule,
    predict_tokens_with_scoring,
)
from mstar.model.omnivoice.config import OmniVoiceConfig

MODEL_PATH = os.environ.get("OMNIVOICE_PATH", "k2-fsa/OmniVoice")
NUM_CODEBOOK = 8
MASK_ID = 1024


def _item(request_id: str, prefix_len: int, target_len: int, guidance_scale: float):
    return CanvasItem(
        request_id=request_id,
        prefix_ids=torch.randint(0, 1000, (NUM_CODEBOOK, prefix_len)),
        prefix_audio_mask=torch.zeros(prefix_len, dtype=torch.bool),
        tokens=torch.full((1, NUM_CODEBOOK, target_len), MASK_ID, dtype=torch.long),
        guidance_scale=guidance_scale,
    )


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def test_canvas_batch_has_no_fully_masked_row():
    """Every query position must see at least one key, padding included."""
    items = [
        _item("a", prefix_len=40, target_len=100, guidance_scale=2.0),
        _item("b", prefix_len=12, target_len=20, guidance_scale=2.0),
        _item("c", prefix_len=31, target_len=55, guidance_scale=0.0),
    ]
    batch = build_canvas_batch(items, MASK_ID, torch.device("cpu"))

    visible = batch.attention_mask.any(dim=-1)  # [R, 1, L]
    assert visible.all(), (
        "a query row with no visible key softmaxes to NaN and contaminates the "
        f"document: {(~visible).sum().item()} such rows"
    )


def test_canvas_batch_row_layout():
    """Conditional rows first, then one unconditional row per CFG request."""
    items = [
        _item("a", prefix_len=40, target_len=100, guidance_scale=2.0),
        _item("b", prefix_len=12, target_len=20, guidance_scale=0.0),
        _item("c", prefix_len=31, target_len=55, guidance_scale=2.0),
    ]
    batch = build_canvas_batch(items, MASK_ID, torch.device("cpu"))

    # 3 conditional + 2 unconditional (item "b" opted out of CFG).
    assert batch.input_ids.shape[0] == 5
    assert [it.cond_row for it in items] == [0, 1, 2]
    assert items[1].uncond_row == -1
    assert items[0].uncond_row == 3 and items[2].uncond_row == 4

    # The longest document sets the pad width: item "a" at 40 + 100.
    assert batch.input_ids.shape[-1] == 140

    # The unconditional document is the target canvas alone.
    assert batch.audio_mask[items[0].uncond_row, :100].all()
    assert not batch.audio_mask[items[0].uncond_row, 100:].any()


def test_reveal_schedule_covers_the_canvas():
    """Whatever the rounding, every cell is revealed exactly once."""
    for num_step in (1, 8, 16, 32, 64):
        for target_len in (1, 7, 137, 750):
            schedule = build_reveal_schedule(
                target_len=target_len,
                num_codebook=NUM_CODEBOOK,
                num_step=num_step,
                t_shift=0.1,
            )
            assert len(schedule) == num_step
            assert min(schedule) >= 0
            assert sum(schedule) == target_len * NUM_CODEBOOK


def test_cfg_off_skips_the_unconditional_slice():
    """guidance_scale 0 must not read a row that was never built."""
    item = _item("solo", prefix_len=10, target_len=6, guidance_scale=0.0)
    batch = build_canvas_batch([item], MASK_ID, torch.device("cpu"))
    logits = torch.randn(batch.input_ids.shape[0], NUM_CODEBOOK, 16, 1025)
    c_logits, u_logits = batch.slice_logits(logits, item)
    assert c_logits.shape == (1, NUM_CODEBOOK, 6, 1025)
    assert u_logits is c_logits

    pred, scores = predict_tokens_with_scoring(
        c_logits, u_logits, MASK_ID, guidance_scale=0.0, class_temperature=0.0
    )
    assert pred.shape == (1, NUM_CODEBOOK, 6)
    assert (pred != MASK_ID).all(), "the MASK class must never be predicted"


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("OMNIVOICE_SKIP_TOKENIZER") == "1",
    reason="tokenizer download disabled",
)
def test_prefix_parity():
    """The canvas prefix must match the reference token for token."""
    pytest.importorskip("omnivoice")
    from omnivoice.models.omnivoice import OmniVoice
    from omnivoice.models.omnivoice import OmniVoiceConfig as RefConfig
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    cases = [
        dict(text="Xin chào, đây là giọng đọc tiếng Việt.", language="Vietnamese",
             instruct=None, ref_text=None, ref_len=0),
        dict(text="Hello [laughter] world.", language="English",
             instruct="a calm elderly man", ref_text=None, ref_len=0),
        dict(text="今天天气很好。", language="Chinese",
             instruct=None, ref_text="你好。", ref_len=37),
    ]

    # _prepare_inference_inputs reads only config, text_tokenizer and device.
    # OmniVoice.__new__ would skip __init__ and leave `.device` -- a
    # PreTrainedModel property over parameters() -- raising, so stand in a
    # plain object carrying the three attributes it actually touches.
    class _Ref:
        config = RefConfig()
        text_tokenizer = tokenizer
        device = torch.device("cpu")

    ref_model = _Ref()

    for case in cases:
        ref_audio_tokens = (
            torch.randint(0, 1024, (NUM_CODEBOOK, case["ref_len"]))
            if case["ref_len"]
            else None
        )
        ours_ids, ours_mask = build_prefix(
            tokenizer=tokenizer,
            text=case["text"],
            num_audio_codebook=NUM_CODEBOOK,
            language=case["language"],
            instruct=case["instruct"],
            ref_text=case["ref_text"],
            ref_audio_tokens=ref_audio_tokens,
            denoise=True,
        )

        target_len = 5
        theirs = OmniVoice._prepare_inference_inputs(
            ref_model,
            text=case["text"],
            num_target_tokens=target_len,
            ref_text=case["ref_text"],
            ref_audio_tokens=ref_audio_tokens,
            lang=case["language"],
            instruct=case["instruct"],
            denoise=True,
        )
        their_prefix = theirs["input_ids"][0, :, :-target_len]
        their_mask = theirs["audio_mask"][0, :-target_len]

        assert torch.equal(ours_ids, their_prefix), f"prefix ids differ for {case['text']!r}"
        assert torch.equal(ours_mask, their_mask), f"audio mask differs for {case['text']!r}"


# ---------------------------------------------------------------------------
# end to end
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_unmask_parity():
    """Fully greedy, both paths, token-exact canvas.

    Batch of one: the reference's dense batching pads conditional rows without
    a self-attending diagonal, so a mixed-length reference batch is not a sound
    comparison target.  Batch-of-one sidesteps that and still exercises every
    line of the ported math.
    """
    pytest.importorskip("omnivoice")
    from omnivoice.models.omnivoice import (
        GenerationTask,
        OmniVoice,
        OmniVoiceGenerationConfig,
    )

    from mstar.model.omnivoice.components.backbone import OmniVoiceBackbone
    from mstar.model.omnivoice.components.unmask import apply_reveal

    config = OmniVoiceConfig()
    gen = dict(num_step=8, guidance_scale=2.0, t_shift=0.1,
               layer_penalty_factor=5.0, position_temperature=0.0,
               class_temperature=0.0)

    reference = OmniVoice.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).eval().cuda()

    text = "Xin chào, đây là một câu thử."
    target_len = 60

    task = GenerationTask(
        batch_size=1, texts=[text], target_lens=[target_len], langs=["Vietnamese"],
        instructs=[None], ref_texts=[None], ref_audio_tokens=[None], ref_rms=[None],
    )
    their_tokens = reference._generate_iterative(
        task, OmniVoiceGenerationConfig(**gen, denoise=True)
    )[0]

    backbone = OmniVoiceBackbone(
        llm=reference.llm,
        audio_embeddings=reference.audio_embeddings,
        audio_heads=reference.audio_heads,
        codebook_layer_offsets=reference.codebook_layer_offsets,
        config=config,
    ).eval()

    prefix_ids, prefix_audio_mask = build_prefix(
        tokenizer=reference.text_tokenizer,
        text=text,
        num_audio_codebook=config.num_audio_codebook,
        language="Vietnamese",
        denoise=True,
    )
    item = CanvasItem(
        request_id="parity",
        prefix_ids=prefix_ids.cuda(),
        prefix_audio_mask=prefix_audio_mask.cuda(),
        tokens=torch.full(
            (1, config.num_audio_codebook, target_len),
            config.audio_mask_id, dtype=torch.long, device="cuda",
        ),
        guidance_scale=gen["guidance_scale"],
    )
    schedule = build_reveal_schedule(
        target_len=target_len,
        num_codebook=config.num_audio_codebook,
        num_step=gen["num_step"],
        t_shift=gen["t_shift"],
    )

    for k in range(gen["num_step"]):
        batch = build_canvas_batch([item], config.audio_mask_id, torch.device("cuda"))
        with torch.inference_mode():
            logits = backbone(
                input_ids=batch.input_ids,
                audio_mask=batch.audio_mask,
                attention_mask=batch.attention_mask,
            ).to(torch.float32)
        c_logits, u_logits = batch.slice_logits(logits, item)
        pred, scores = predict_tokens_with_scoring(
            c_logits, u_logits, config.audio_mask_id,
            guidance_scale=gen["guidance_scale"],
            class_temperature=gen["class_temperature"],
        )
        apply_reveal(
            tokens=item.tokens, pred_tokens=pred, scores=scores,
            reveal_count=schedule[k], audio_mask_id=config.audio_mask_id,
            layer_penalty_factor=gen["layer_penalty_factor"],
            position_temperature=gen["position_temperature"],
        )

    ours = item.tokens[0].cpu()
    assert (ours != config.audio_mask_id).all(), "canvas left masked cells"
    mismatch = int((ours != their_tokens.cpu()).sum())
    assert mismatch == 0, (
        f"{mismatch}/{ours.numel()} cells differ from the reference canvas"
    )
