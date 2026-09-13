# msao-exp — OmniVoice served on M*, for A/B against misa-tts-omnivoice-api-*.
#
# Base is the sglang image already in aiteam, chosen because it happens to
# carry every hard dependency at a version this needs:
#
#   torch 2.13.0+cu129   transformers 5.12.1 (>= 5.3, ships HiggsAudioV2Tokenizer)
#   flashinfer 0.6.18    (>= 0.6.15, the ragged-attention wrapper OmniVoice plans)
#
# Using it also means the push is a thin layer: the base already lives in the
# registry, so only this image's own layers travel.
#
# mstar installs with --no-deps ON PURPOSE. Its pyproject pins torch <2.13.0
# and the base has exactly 2.13.0, so a dependency resolve would downgrade
# torch — which would break the flashinfer build's ABI and defeat the reason
# for picking this base. The deps that are genuinely missing are listed
# explicitly below instead.
FROM 10.8.28.92/aiteam/sglang-tmduc:cu12-20260910b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# TWO filesystem layers, total. This is a hard budget, not a style choice.
#
# The base sits at 121 layers and Docker's ceiling is 125. The first build of
# this image came to 127 and the registry mirror rejected it outright with
# "failed to register layer: max depth exceeded"; a three-layer version landed
# on exactly 125, i.e. no headroom at all. So the source is copied FIRST and
# everything else happens in one RUN, leaving two layers spare.
#
# The cost is cache behaviour: touching any source file invalidates the apt and
# pip work below. That is the wrong trade in general and the right one here --
# a rebuild is minutes, a mirror that refuses the image is a dead end.
#
# The durable fix is a thinner base. This one is a 50 GB sglang *development*
# image and OmniVoice is a 0.6B model; it was chosen because it already carried
# torch 2.13.0+cu129, transformers 5.12.1 and flashinfer 0.6.18 at compatible
# versions, which is worth a lot, but 121 layers of someone else's build steps
# is a ceiling this image will hit again.
COPY . /opt/mstar
WORKDIR /opt/mstar

# What this single layer does:
#   - drops the base's two dead NVIDIA apt sources (devtools 403s, CUDA fails
#     certificate verification); either one aborts apt-get update, and nothing
#     installed here comes from them
#   - ffmpeg, which backs pydub for the reference's silence trim on the decode tail
#   - omnivoice from git at a pinned commit, NOT PyPI: the published 0.2.1 wheel
#     does not carry omnivoice/models/omnivoice_flashinfer.py, the packed
#     fused-attention path this integration is built on. The pin is the exact
#     tree the port was written and reviewed against.
#   - --no-deps throughout: mstar and omnivoice both pin torch ranges that
#     exclude the base's 2.13.0, and a resolve would downgrade torch and break
#     flashinfer's ABI, defeating the reason for choosing this base. torchaudio,
#     soundfile and torchcodec are already present, built against that torch.
#   - qwen-tts, for the Qwen3-TTS 12 Hz codec. --no-deps is required, not
#     just preferred: its metadata pins transformers==4.57.3 against the
#     base's 5.12.1. Only the files are needed — qwen3_tts_model loads the
#     two tokenizer_12hz sources directly under a private package name so
#     the package's own __init__ (which drags in pysox) never runs. The
#     published 0.1.1 wheel ships that directory WITHOUT an __init__.py —
#     it is a namespace package there — while the loader does
#     spec_from_file_location(..., source_root / "__init__.py"), so an
#     empty one is created here. Safe: both modules import only torch,
#     transformers and numpy, with no relative imports to resolve.
#
#     The transformers pin is not arbitrary either: the codec source is
#     written against 4.57's `check_model_inputs()`, a decorator factory.
#     In 5.x the same name IS the decorator and takes the function, so the
#     call form raises TypeError at import. Rewriting the one call site to
#     `@check_model_inputs` restores the 4.57 default behaviour under the
#     5.x API. The guard below then loads the classes through M*'s own
#     loader, i.e. the exact path that failed at serving time.
#
#     Same story one layer down: the codec's rotary embedding looks up
#     ROPE_INIT_FUNCTIONS['default'], a key 5.x no longer registers (it
#     dropped _compute_default_rope_parameters outright). The shim
#     re-registers it with the textbook definition, inv_freq =
#     1 / base**(2i/d) with unit scaling — that is what 'default' always
#     meant, not a reconstruction of anything version-specific.
#   - accelerate, which is NOT optional despite --no-deps. transformers'
#     from_pretrained calls check_and_set_device_map(), and that raises as
#     soon as a torch device context is active -- which it is, because the
#     engine manager builds submodules under one. Leaving it out got as far
#     as a running pod before dying in HiggsAudioV2TokenizerModel.from_pretrained.
#   - four guards that fail the BUILD rather than the first request: the
#     private omnivoice surface the backbone drives, the model registry entry,
#     and the backbone's own API check.
RUN rm -f /etc/apt/sources.list.d/*cuda* /etc/apt/sources.list.d/*nvidia* \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir --no-deps \
      "omnivoice @ git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2" \
      "qwen-tts==0.1.1" \
 && python3 -c "import importlib.metadata as m, pathlib; \
p = pathlib.Path(m.distribution('qwen-tts').locate_file('qwen_tts/core/tokenizer_12hz')); \
(p / '__init__.py').touch(); \
f = p / 'modeling_qwen3_tts_tokenizer_v2.py'; t = f.read_text(); \
assert '@check_model_inputs()' in t, 'decorator call not found; check the wheel'; \
t = t.replace('@check_model_inputs()', '@check_model_inputs'); \
anchor = 'from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update'; \
assert anchor in t, 'rope import not found; check the wheel'; \
t = t.replace(anchor, anchor + chr(10) + open('/opt/mstar/docker/qwen_tts_rope_default.py').read()); \
f.write_text(t); \
print('qwen-tts 12hz: __init__.py added, decorator and rope default adapted')" \
 && pip install --no-cache-dir pydub num2words accelerate \
 && pip install --no-cache-dir --no-deps -e . \
 && python3 -c "import omnivoice.models.omnivoice_flashinfer as fi; \
[getattr(fi, n) for n in ('_CTX','PackedAttnRunner','_forward_logits','apply_flashinfer')]; \
print('omnivoice + flashinfer surface ok')" \
 && python3 -c "from mstar.model.registry import get_model_class; \
print('registry:', get_model_class('omnivoice').__name__, get_model_class('qwen3_tts').__name__, get_model_class('qwen3_asr').__name__)" \
 && python3 -c "from mstar.model.qwen3_tts.qwen3_tts_model import _load_qwen3_tts_decoder_classes as L; \
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as R; \
cls = L(); assert 'default' in R, sorted(R); \
print('qwen-tts 12hz codec loads:', [c.__name__ for c in cls], '| rope default registered')" \
 && python3 -c "from mstar.model.omnivoice.components.backbone import assert_flashinfer_api; \
assert_flashinfer_api(); print('backbone api guard ok')" \
 && python3 -c "import torch; assert torch.__version__.startswith('2.13.'), torch.__version__; \
print('torch intact:', torch.__version__)" \
 && python3 -c "import accelerate, transformers, omnivoice; \
from mstar.model.omnivoice import omnivoice_model, submodules; \
from mstar.model.omnivoice.components import backbone, codec, text, unmask; \
print('serving imports ok; accelerate', accelerate.__version__)"

EXPOSE 8080

# Run:AI serves a workload under /<project>/<job-name>/ and does not strip the
# prefix, so the spec sets MSTAR_ROOT_PATH to the same value it gives
# APP_ROOT_PATH. Empty here: a direct `docker run` is served at /.
ENV MSTAR_ROOT_PATH="" \
    OMNIVOICE_CHECKPOINT_DIR=/weights/omnivoice

# The entrypoint syncs the checkpoint from S3 when the spec names one, then
# execs the serve command after `--`.
ENTRYPOINT ["python3", "/opt/mstar/docker/msao-entrypoint.py", "--"]
CMD ["mstar", "serve", "omnivoice", \
     "--host", "0.0.0.0", "--port", "8080", \
     "--config", "/opt/mstar/configs/omnivoice.yaml", \
     "--log-stats"]
