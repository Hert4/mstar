"""FastAPI routes for the OpenAI-compatible API.

Endpoints stay model-agnostic: each looks up the loaded model's adapter, checks
the surface is supported, and delegates to a serving handler. The native
``/generate`` endpoint is unaffected.
"""

from __future__ import annotations

import base64
import json

from fastapi import APIRouter, Request
from pydantic import ValidationError
from fastapi.responses import JSONResponse, StreamingResponse

from mstar.api_server.openai import (
    serving_chat,
    serving_images,
    serving_speech,
    serving_videos,
)
from mstar.api_server.openai._util import now
from mstar.api_server.openai.adapters import get_adapter
from mstar.api_server.openai.protocol import (
    ChatCompletionRequest,
    ImageGenerationRequest,
    ModelCard,
    ModelList,
    SpeechRequest,
    VideoGenerationRequest,
)

router = APIRouter()


def _api():
    # The server runs either as the package module (console scripts ``mstar`` /
    # ``mstar-serve``) or as ``__main__`` (``python mstar/api_server/entrypoint.py``,
    # used by the test/*/launch_server_*.sh scripts). main() sets ``api_server``
    # on whichever module it runs in, so resolve the live instance from both —
    # via sys.modules (importing would re-execute the entrypoint module).
    import sys

    for name in ("mstar.api_server.entrypoint", "__main__"):
        mod = sys.modules.get(name)
        api = getattr(mod, "api_server", None) if mod is not None else None
        if api is not None:
            return api
    return None


def _error(status: int, message: str, type_: str = "invalid_request_error") -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={"error": {"message": message, "type": type_, "code": status}},
    )


def _resolve(require: str):
    """Return (api, model_name, adapter, error_response). ``error`` is non-None
    when the loaded model can't serve ``require`` (e.g. 'supports_chat')."""
    api = _api()
    if api is None:
        return None, None, None, _error(503, "Server not ready", "server_error")
    adapter = get_adapter(api.model_name)
    if adapter is None:
        return api, api.model_name, None, _error(
            404, f"Model {api.model_name!r} has no OpenAI-compatible adapter; use POST /generate", "model_not_found"
        )
    if not getattr(adapter, require, False):
        return api, api.model_name, adapter, _error(
            404, f"Model {api.model_name!r} does not support this endpoint"
        )
    return api, api.model_name, adapter, None


@router.get("/v1/models")
async def list_models():
    api = _api()
    name = api.model_name if api is not None else "unknown"
    return JSONResponse(ModelList(data=[ModelCard(id=name, created=now())]).model_dump())


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw_request: Request):
    api, model_name, adapter, err = _resolve("supports_chat")
    if err is not None:
        return err
    try:
        result = await serving_chat.create_chat_completion(api, model_name, adapter, request, raw_request)
    except Exception as e:  # noqa: BLE001 — surface as an OpenAI error envelope
        default_status = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return _error(getattr(e, "status_code", default_status), str(getattr(e, "detail", e)), "server_error")
    if request.stream:
        return StreamingResponse(
            result, media_type="text/event-stream", headers={"Cache-Control": "no-cache"}
        )
    return JSONResponse(result)


# ---------------------------------------------------------------------------
# Compatibility with the in-house TTS wrapper this endpoint replaces.
#
# That service speaks its own multipart dialect -- `text` / `texts`,
# `encode_type`, `voice_type`, ISO language codes -- and there are clients in
# production written against it. Accepting its spelling costs one dict and one
# function here, and lets those clients change only the URL. The OpenAI names
# stay canonical: if both are present the OpenAI one wins.
# ---------------------------------------------------------------------------

# The wrapper sends ISO codes; OmniVoice is prompted with the language's name
# (it goes into the prompt verbatim as <|lang_start|>...<|lang_end|>, so an
# unknown code would be passed through to the model as-is and quietly skew the
# voice). Same mapping the ASR model uses for its own `language` field.
_WRAPPER_LANG = {
    "vi": "Vietnamese", "en": "English", "zh": "Chinese", "yue": "Cantonese",
    "ja": "Japanese", "ko": "Korean", "th": "Thai", "ar": "Arabic",
    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "id": "Indonesian", "ms": "Malay",
    "hi": "Hindi", "tr": "Turkish", "nl": "Dutch", "pl": "Polish",
}

_WRAPPER_ALIASES = {"text": "input", "encode_type": "response_format"}


def _apply_wrapper_aliases(data: dict) -> dict:
    """Translate the wrapper's field names in place. OpenAI names take priority."""
    for old, new in _WRAPPER_ALIASES.items():
        if old in data and new not in data:
            data[new] = data.pop(old)
        else:
            data.pop(old, None)

    # Deliberately NOT aliased to `voice`. The wrapper's voice_type names a
    # preset in its own ref_voices/ directory; this server has no such registry
    # and `voice` is a free-text description of a voice to design. Feeding a
    # preset's name in as a description returns a different voice and no error,
    # so say what is wrong instead.
    if data.pop("voice_type", None):
        raise ValueError(
            "voice_type names a preset this server does not have. Clone a voice "
            "with ref_audio (a file, data URL, path or URL), or describe one "
            "with voice."
        )

    # `texts` is a JSON array of strings in one field, which is how the wrapper
    # spells a batch. Bad JSON here is the caller's mistake, so let it surface
    # as a 400 rather than silently synthesising the literal string.
    if "texts" in data:
        raw = data.pop("texts")
        if "input" not in data:
            if isinstance(raw, str):
                parsed = json.loads(raw)
                if not isinstance(parsed, list):
                    raise ValueError("texts must be a JSON array of strings")
                data["input"] = parsed
            else:
                data["input"] = raw

    lang = data.get("language")
    if isinstance(lang, str) and lang.lower() in _WRAPPER_LANG:
        data["language"] = _WRAPPER_LANG[lang.lower()]
    return data


async def _speech_request(raw_request: Request) -> SpeechRequest:
    """Build a SpeechRequest from either a JSON body or a multipart form.

    The JSON body is the OpenAI shape. Multipart is accepted too because this
    endpoint takes an audio file: ``-F ref_audio=@voice.wav`` beats pasting a
    200KB data URL into the body, and it matches how OpenAI's own audio
    endpoints that carry a file (transcriptions, translations) are called.
    An uploaded file is folded into the data URL the adapter already resolves.
    """
    ctype = raw_request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        body = await raw_request.json()
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")
        return SpeechRequest.model_validate(_apply_wrapper_aliases(body))

    form = await raw_request.form()
    data: dict = {}
    for key, value in form.multi_items():
        if hasattr(value, "read"):
            raw = await value.read()
            if not raw:
                continue
            mime = getattr(value, "content_type", None) or "audio/wav"
            value = f"data:{mime};base64,{base64.b64encode(raw).decode()}"
        # A repeated field is a list, so `-F input=one -F input=two` is the
        # multipart spelling of the JSON list form.
        if key in data:
            prev = data[key]
            data[key] = [*prev, value] if isinstance(prev, list) else [prev, value]
        else:
            data[key] = value
    return SpeechRequest.model_validate(_apply_wrapper_aliases(data))


@router.post("/v1/audio/speech")
async def audio_speech(raw_request: Request):
    try:
        request = await _speech_request(raw_request)
    except ValidationError as e:
        detail = "; ".join(
            f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        return _error(422, detail or str(e), "invalid_request_error")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as e:
        return _error(400, f"malformed request body: {e}", "invalid_request_error")
    api, model_name, adapter, err = _resolve("supports_speech")
    if err is not None:
        return err
    try:
        return await serving_speech.create_speech(api, model_name, adapter, request, raw_request)
    except Exception as e:  # noqa: BLE001 — surface as an OpenAI error envelope
        default_status = 400 if isinstance(e, (ValueError, TypeError)) else 500
        status = getattr(e, "status_code", default_status)
        kind = "invalid_request_error" if status < 500 else "server_error"
        return _error(status, str(getattr(e, "detail", e)), kind)


@router.post("/v1/images/generations")
async def images_generations(request: ImageGenerationRequest, raw_request: Request):
    api, model_name, adapter, err = _resolve("supports_images")
    if err is not None:
        return err
    try:
        result = await serving_images.create_images(api, model_name, adapter, request, raw_request)
    except Exception as e:  # noqa: BLE001
        return _error(getattr(e, "status_code", 500), str(getattr(e, "detail", e)), "server_error")
    return JSONResponse(result)


@router.post("/v1/videos/generations")
async def videos_generations(request: VideoGenerationRequest):
    api, model_name, adapter, err = _resolve("supports_videos")
    if err is not None:
        return err
    try:
        result = await serving_videos.create_videos(api, model_name, adapter, request)
    except Exception as e:  # noqa: BLE001
        default_status = 400 if isinstance(e, (ValueError, TypeError)) else 500
        return _error(getattr(e, "status_code", default_status), str(getattr(e, "detail", e)), "server_error")
    return JSONResponse(result)


@router.post("/v1/images/edits")
async def images_edits(request: Request):
    # Multipart (image file + prompt + passthrough fields), parsed manually so
    # arbitrary model knobs (e.g. cfg_*_scale) flow through as model_kwargs.
    api, model_name, adapter, err = _resolve("supports_images")
    if err is not None:
        return err
    try:
        form = await request.form()
        image = form.get("image")
        if image is None or not hasattr(image, "read"):
            return _error(400, "images/edits requires an 'image' file upload")
        image_bytes = await image.read()
        prompt = form.get("prompt") or ""
        known = {"image", "prompt", "model", "n", "size", "response_format"}
        extra: dict = {}
        for key, value in form.multi_items():
            if key in known or hasattr(value, "filename"):
                continue
            try:
                extra[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                extra[key] = value
        result = await serving_images.create_image_edit(
            api,
            model_name,
            adapter,
            prompt=prompt,
            image_bytes=image_bytes,
            image_filename=getattr(image, "filename", None),
            model_kwargs=extra,
            raw_request=request,
        )
    except Exception as e:  # noqa: BLE001
        return _error(getattr(e, "status_code", 500), str(getattr(e, "detail", e)), "server_error")
    return JSONResponse(result)
