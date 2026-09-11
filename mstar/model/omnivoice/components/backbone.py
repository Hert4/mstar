"""The OmniVoice backbone: a Qwen3 body read bidirectionally over a token canvas.

Three things live here:

``OmniVoiceBackbone``
    The checkpoint's modules — the Qwen3 body plus the 8-codebook audio
    embedding table and the audio head — and the embedding merge that turns a
    ``[B, C, S]`` canvas into ``[B, S, hidden]``.  A port of
    ``OmniVoice._prepare_embed_inputs`` / ``OmniVoice.forward``.

``CanvasItem`` / ``build_canvas_batch``
    The dense padded batch a step runs on.  This is what lets one forward serve
    several concurrent requests, which a fixed single-request pipeline cannot.

The Qwen3 body itself is the stock HuggingFace module loaded from the
checkpoint, not a re-implementation.  wan22's text_encoder node takes the same
approach with ``UMT5EncoderModel``: parity comes free, and the value M* adds
here is the graph, the loop, the batching and the per-request knobs rather than
a second copy of Qwen3.  Porting the body onto ``mstar.model.components`` is
what unlocks TP and CUDA graphs, and is deliberately left to a follow-up — it
buys throughput, not correctness, and it should not gate a first working serve.
"""

import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from mstar.model.omnivoice.config import OmniVoiceConfig

logger = logging.getLogger(__name__)


class OmniVoiceBackbone(nn.Module):
    """Qwen3 body + audio embedding table + audio head.

    Attention is bidirectional and there is no KV cache: the target region is
    rewritten every diffusion step, and since the prefix attends *into* that
    region its hidden states change with it.  Nothing survives a step, which is
    why the model declares no cache resource.
    """

    def __init__(
        self,
        llm: nn.Module,
        audio_embeddings: nn.Embedding,
        audio_heads: nn.Linear,
        codebook_layer_offsets: torch.Tensor,
        config: OmniVoiceConfig,
    ):
        super().__init__()
        self.llm = llm
        self.audio_embeddings = audio_embeddings
        self.audio_heads = audio_heads
        self.register_buffer("codebook_layer_offsets", codebook_layer_offsets, persistent=False)
        self.config = config

    def prepare_embed_inputs(
        self, input_ids: torch.Tensor, audio_mask: torch.Tensor
    ) -> torch.Tensor:
        """``[B, C, S]`` ids + ``[B, S]`` audio mask -> ``[B, S, hidden]``.

        Text positions read row 0 through the body's own embedding table; audio
        positions sum one embedding per codebook row, each row offset into its
        own block of the table.  The ``* audio_mask`` before the offset add is
        the reference's: it keeps text positions from indexing out of the audio
        table even though ``torch.where`` discards their value anyway.
        """
        text_embeds = self.llm.get_input_embeddings()(input_ids[:, 0, :])

        shifted_ids = (
            input_ids * audio_mask.unsqueeze(1)
        ) + self.codebook_layer_offsets.view(1, -1, 1)
        audio_embeds = self.audio_embeddings(shifted_ids).sum(dim=1)

        return torch.where(audio_mask.unsqueeze(-1), audio_embeds, text_embeds)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """One bidirectional pass; returns audio logits ``[B, C, S, V]``."""
        inputs_embeds = self.prepare_embed_inputs(input_ids, audio_mask)
        hidden_states = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )[0]

        batch_size, seq_len, _ = hidden_states.shape
        logits_flat = self.audio_heads(hidden_states)
        return logits_flat.view(
            batch_size,
            seq_len,
            self.config.num_audio_codebook,
            self.config.audio_vocab_size,
        ).permute(0, 2, 1, 3)


@dataclass
class CanvasItem:
    """One request's contribution to a step's batch.

    ``prefix_ids`` is ``[C, N]`` — style tokens, then text tokens, then the
    reference audio tokens when cloning.  ``tokens`` is ``[1, C, T]``, the live
    target canvas: all MASK at step 0, fully revealed when the loop ends.
    """

    request_id: str
    prefix_ids: torch.Tensor
    prefix_audio_mask: torch.Tensor
    tokens: torch.Tensor
    guidance_scale: float
    # Filled in by build_canvas_batch; the scoring step reads them back.
    cond_row: int = field(default=-1)
    uncond_row: int = field(default=-1)

    @property
    def target_len(self) -> int:
        return self.tokens.shape[-1]

    @property
    def cond_len(self) -> int:
        return self.prefix_ids.shape[-1] + self.target_len

    @property
    def does_cfg(self) -> bool:
        return self.guidance_scale != 0


@dataclass
class CanvasBatch:
    """Dense padded inputs for one backbone forward."""

    input_ids: torch.Tensor       # [R, C, L]
    audio_mask: torch.Tensor      # [R, L]
    attention_mask: torch.Tensor  # [R, 1, L, L]
    items: list[CanvasItem]

    def slice_logits(
        self, logits: torch.Tensor, item: CanvasItem
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The conditional and unconditional target logits for one item.

        The target sits at the end of the conditional document, and the
        unconditional document is that target alone, so the two slices are
        ``[c_len - t : c_len]`` and ``[:t]`` respectively.  With CFG off the
        unconditional slice is never read, and the conditional one is returned
        twice rather than allocating a dummy.
        """
        t = item.target_len
        c_len = item.cond_len
        c_logits = logits[item.cond_row : item.cond_row + 1, :, c_len - t : c_len, :]
        if not item.does_cfg:
            return c_logits, c_logits
        u_logits = logits[item.uncond_row : item.uncond_row + 1, :, :t, :]
        return c_logits, u_logits


def build_canvas_batch(
    items: list[CanvasItem],
    audio_mask_id: int,
    device: torch.device,
) -> CanvasBatch:
    """Pack a step's requests into one padded batch.

    Layout follows the reference: every conditional document first, then every
    unconditional one, so a single forward covers both halves of CFG for the
    whole batch.  Items that disable CFG contribute no unconditional row.

    Padding runs to the batch's longest document.  Pad positions carry the MASK
    id and, critically, get a **self-attending diagonal** so no query row is
    fully masked.  A fully-masked row softmaxes to NaN, and because a masked key
    still contributes ``0 * NaN`` to every other query's weighted sum, one such
    row silently poisons the whole document on the next layer.  The diagonal
    costs nothing and changes no consumed position: pad rows are never read, and
    valid queries already mask pad keys out.  (The reference applies this fix to
    its unconditional rows only; whether its conditional rows trip the hazard on
    a mixed-length batch is worth confirming on real weights and reporting
    upstream if so.)
    """
    if not items:
        raise ValueError("build_canvas_batch called with no items")

    num_codebook = items[0].prefix_ids.shape[0]
    cond_items = items
    cfg_items = [it for it in items if it.does_cfg]

    lengths = [it.cond_len for it in cond_items] + [it.target_len for it in cfg_items]
    max_len = max(lengths)
    num_rows = len(cond_items) + len(cfg_items)

    input_ids = torch.full(
        (num_rows, num_codebook, max_len), audio_mask_id, dtype=torch.long, device=device
    )
    audio_mask = torch.zeros((num_rows, max_len), dtype=torch.bool, device=device)
    attention_mask = torch.zeros(
        (num_rows, 1, max_len, max_len), dtype=torch.bool, device=device
    )

    def _fill(row: int, ids: torch.Tensor, amask: torch.Tensor) -> None:
        length = ids.shape[-1]
        input_ids[row, :, :length] = ids
        audio_mask[row, :length] = amask
        attention_mask[row, :, :length, :length] = True
        if length < max_len:
            pad = torch.arange(length, max_len, device=device)
            attention_mask[row, :, pad, pad] = True

    for row, item in enumerate(cond_items):
        item.cond_row = row
        item.uncond_row = -1
        ids = torch.cat(
            [item.prefix_ids.to(device), item.tokens[0].to(device)], dim=-1
        )
        amask = torch.cat(
            [
                item.prefix_audio_mask.to(device),
                torch.ones(item.target_len, dtype=torch.bool, device=device),
            ],
            dim=-1,
        )
        _fill(row, ids, amask)

    for offset, item in enumerate(cfg_items):
        row = len(cond_items) + offset
        item.uncond_row = row
        # The unconditional document is the target canvas alone — no style, no
        # text, no reference. Dropping the conditioning *is* the null prompt.
        _fill(
            row,
            item.tokens[0].to(device),
            torch.ones(item.target_len, dtype=torch.bool, device=device),
        )

    return CanvasBatch(
        input_ids=input_ids,
        audio_mask=audio_mask,
        attention_mask=attention_mask,
        items=items,
    )
