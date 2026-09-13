
# --- injected by msao-exp.Dockerfile -------------------------------------
# transformers 5.x dropped the 'default' entry from ROPE_INIT_FUNCTIONS (and
# _compute_default_rope_parameters with it), while this file still asks for it
# by name whenever a config carries no rope_scaling. Re-register it with the
# textbook definition so the lookup below resolves.
if "default" not in ROPE_INIT_FUNCTIONS:
    import torch as _torch

    def _mstar_default_rope(config, device=None, seq_len=None, **rope_kwargs):
        base = config.rope_theta
        dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base
            ** (
                _torch.arange(0, dim, 2, dtype=_torch.int64).to(
                    device=device, dtype=_torch.float
                )
                / dim
            )
        )
        return inv_freq, 1.0

    ROPE_INIT_FUNCTIONS["default"] = _mstar_default_rope
# -------------------------------------------------------------------------
