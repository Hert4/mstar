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

# ffmpeg backs pydub, which the reference's silence trim uses on the decode tail.
#
# The base ships two NVIDIA apt sources that no longer resolve -- the devtools
# list 403s and the CUDA one fails certificate verification -- and either kills
# `apt-get update` outright. They are dropped rather than worked around with
# [trusted=yes]: nothing installed here comes from them.
RUN rm -f /etc/apt/sources.list.d/*cuda* /etc/apt/sources.list.d/*nvidia* \
 && apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/*

# The reference package: mstar reuses its weight loading, its rule-based
# duration estimator, its audio post-processing and — the reason performance
# matches — its fused packed-attention forward. --no-deps for the torch reason
# above; its own pure-python requirements come next.
# From git at a pinned commit, NOT from PyPI: the published omnivoice 0.2.1
# wheel does not carry omnivoice/models/omnivoice_flashinfer.py, which is the
# packed fused-attention path this integration is built on. The pin is the
# exact tree the port was written and reviewed against.
#
# torchaudio, soundfile and torchcodec are already in the base at versions
# built against its torch; installing them again risks pulling a mismatched
# pair, so only the two genuinely missing pure-python deps are named.
RUN pip install --no-cache-dir --no-deps \
      "omnivoice @ git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2" \
 && pip install --no-cache-dir pydub num2words \
 && python3 -c "import omnivoice, omnivoice.models.omnivoice_flashinfer as fi; \
print('omnivoice ok'); \
[getattr(fi, n) for n in ('_CTX','PackedAttnRunner','_forward_logits','apply_flashinfer')]; \
print('flashinfer surface ok')"

WORKDIR /opt/mstar
COPY . /opt/mstar
RUN pip install --no-cache-dir --no-deps -e . \
 && python3 -c "from mstar.model.registry import get_model_class; \
print('registry:', get_model_class('omnivoice').__name__)"

# Fail at build time, not at the first request, if the private surface this
# integration rides on has moved in the omnivoice version that got installed.
RUN python3 -c "from mstar.model.omnivoice.components.backbone import assert_flashinfer_api; \
assert_flashinfer_api(); print('backbone api guard ok')"

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
