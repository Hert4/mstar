#!/usr/bin/env python3
"""Fetch the OmniVoice checkpoint from S3, then hand off to ``mstar serve``.

Run:AI's ``storage.mode: s3`` injects ``AWS_*`` credentials but mounts nothing
for a custom engine, so the container fetches its own weights — the same shape
the baseline ``tts-api-cloud-omnivoice`` image uses, which is what keeps an A/B
against it honest: same checkpoint, same source.

Env:
    OMNIVOICE_CHECKPOINT_S3_URI   s3://bucket/prefix to sync (optional; without
                                  it the model falls back to the public
                                  k2-fsa/OmniVoice weights over the Hub)
    OMNIVOICE_CHECKPOINT_DIR      where to put them (default /weights/omnivoice)
    AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY

Everything after ``--`` is the serve command; arguments are passed through
untouched so the workload spec stays the single place that sets them.
"""

import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

# Weights are re-fetched only when the marker is absent, so a pod restart on a
# warm volume skips a multi-GB download. The marker is written last, so an
# interrupted sync is retried rather than mistaken for a complete one.
_DONE_MARKER = ".msao-fetch-complete"


def _log(msg: str) -> None:
    print(f"[msao-entrypoint] {msg}", flush=True)


def fetch_checkpoint(uri: str, dest: Path) -> None:
    import boto3
    from botocore.config import Config

    parsed = urlparse(uri)
    bucket, prefix = parsed.netloc, parsed.path.lstrip("/")

    endpoint = os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("S3_ENDPOINT")
    if endpoint and not endpoint.startswith(("http://", "https://")):
        # The workload spec carries a bare host; boto3 needs a scheme. The
        # cluster's Ceph gateway is plain HTTP inside the network.
        endpoint = f"http://{endpoint}"

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID")
        or os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        or os.environ.get("S3_SECRET_KEY"),
        config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )

    dest.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    total = 0
    started = time.time()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(prefix) :].lstrip("/")
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, key, str(target))
            total += obj["Size"]
            _log(f"  {rel}  {obj['Size'] / 1e6:.1f} MB")

    if total == 0:
        raise RuntimeError(
            f"No objects under {uri}. Check the bucket, the prefix and the "
            "credentials before assuming the network is at fault."
        )
    elapsed = time.time() - started
    _log(f"fetched {total / 1e9:.2f} GB in {elapsed:.0f}s -> {dest}")
    (dest / _DONE_MARKER).write_text("ok\n")


def main() -> None:
    argv = sys.argv[1:]
    if "--" in argv:
        serve_cmd = argv[argv.index("--") + 1 :]
    else:
        serve_cmd = argv or ["mstar", "serve", "omnivoice"]

    uri = os.environ.get("OMNIVOICE_CHECKPOINT_S3_URI", "").strip()
    dest = Path(os.environ.get("OMNIVOICE_CHECKPOINT_DIR", "/weights/omnivoice"))

    if uri:
        if (dest / _DONE_MARKER).exists():
            _log(f"checkpoint already present at {dest}, skipping fetch")
        else:
            _log(f"fetching {uri}")
            fetch_checkpoint(uri, dest)
        # Point the model at the fetched directory. Set here rather than in the
        # spec so the two can never disagree about where the weights landed.
        os.environ["MSTAR_OMNIVOICE_MODEL_PATH"] = str(dest)
    else:
        _log(
            "OMNIVOICE_CHECKPOINT_S3_URI not set; falling back to the public "
            "k2-fsa/OmniVoice weights over the Hub"
        )

    _log(f"exec: {' '.join(serve_cmd)}")
    os.execvp(serve_cmd[0], serve_cmd)


if __name__ == "__main__":
    main()
