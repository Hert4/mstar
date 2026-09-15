"""/v1/audio/speech handler (text-to-speech).

Non-streaming returns the full audio as a container blob (WAV by default).
Streaming returns a single open-ended WAV response (header + PCM16 frames) as
the audio is produced.
"""

from __future__ import annotations

import asyncio
import base64
import os
import time

from fastapi.responses import JSONResponse, Response, StreamingResponse

from mstar.api_server import media_io
from mstar.api_server.openai._util import rid


# One request per item, so a runaway list cannot pin the scheduler. The
# wrapper service this replaces caps at 16; keep that until there is a reason.
MAX_BATCH = int(os.environ.get("MSTAR_SPEECH_MAX_BATCH", "16"))


async def _one(api, adapter, req, text, fmt, sample_rate, raw_request):
    """Submit a single item and return its encoded audio. Submission is
    synchronous, so the caller submits every item before awaiting any of
    them -- that is what lets the scheduler see them as one batch."""
    args = adapter.speech_to_request(req.model_copy(update={"input": text}), api.upload_dir)
    request_id = rid("speech")
    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        streaming=False,
        request_id=request_id,
    )

    async def _collect():
        chunks = await api.collect_results(request_id, raw_request)
        pcm = b"".join(c.data for c in chunks if c.modality == "audio")
        audio, _mime = media_io.pcm16_to_container(pcm, sample_rate, fmt)
        return audio

    return _collect()


async def _create_speech_batch(api, model_name, adapter, req, raw_request, fmt, sample_rate):
    texts = list(req.input)
    if not texts:
        raise ValueError("input list is empty")
    if len(texts) > MAX_BATCH:
        raise ValueError(f"input list has {len(texts)} items; the limit is {MAX_BATCH}")
    if req.stream:
        raise ValueError("stream is not supported with a list input; send one string to stream")

    # One reference voice covers the whole list, so resolve it once. Left to
    # the per-item path it would be decoded and written to upload_dir once per
    # item -- sixteen copies of the same WAV for a sixteen-line batch. A local
    # path resolves to itself, so the items below just reuse this one.
    ref = (req.model_extra or {}).get("ref_audio")
    if isinstance(ref, str) and ref:
        _mod, ref_path = media_io.resolve_media_ref(ref, api.upload_dir, allow_remote=True)
        req = req.model_copy(update={"ref_audio": ref_path})

    started = time.monotonic()
    # Submit all, then await all: the whole point of the list form.
    pending = [await _one(api, adapter, req, t, fmt, sample_rate, raw_request) for t in texts]
    audios = await asyncio.gather(*pending)
    return JSONResponse({
        "object": "list",
        "model": model_name,
        "created": int(time.time()),
        "inference_time_s": round(time.monotonic() - started, 3),
        "data": [
            {
                "object": "audio.speech",
                "index": i,
                "output_format": fmt,
                "b64_json": base64.b64encode(audio).decode("ascii"),
            }
            for i, audio in enumerate(audios)
        ],
    })


async def create_speech(api, model_name, adapter, req, raw_request=None):  # noqa: ARG001
    if isinstance(req.input, list):
        fmt_ = (req.response_format or "wav").lower()
        rate_ = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
        return await _create_speech_batch(api, model_name, adapter, req, raw_request, fmt_, rate_)

    args = adapter.speech_to_request(req, api.upload_dir)
    request_id = rid("speech")
    sample_rate = api.model.get_output_sample_rate("audio") if api.model is not None else 24000
    fmt = (req.response_format or "wav").lower()

    api.submit_request(
        text=args.text,
        file_paths=args.file_paths,
        input_modalities=args.input_modalities,
        output_modalities=args.output_modalities,
        model_kwargs=args.model_kwargs,
        streaming=bool(req.stream),
        request_id=request_id,
    )

    if req.stream:
        return StreamingResponse(
            _stream_wav(api, request_id, sample_rate),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"},
        )

    chunks = await api.collect_results(request_id, raw_request)
    pcm = b"".join(c.data for c in chunks if c.modality == "audio")
    audio_bytes, mime = media_io.pcm16_to_container(pcm, sample_rate, fmt)
    return Response(content=audio_bytes, media_type=mime)


async def _stream_wav(api, request_id, sample_rate):
    yield media_io.wav_stream_header(sample_rate)
    async for c in api.iter_result_chunks(request_id):
        if c.modality == "audio" and c.data:
            yield c.data
