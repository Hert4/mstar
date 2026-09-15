# OmniVoice speech

`POST /v1/audio/speech` synthesises speech, clones a voice from a reference
recording, and takes a batch of lines in one call. It speaks JSON or
`multipart/form-data`, and it answers to the field names of the in-house TTS
wrapper it replaces.

Every example below is a complete `curl`. Replace `$BASE` with the server root;
on Run:AI that includes the project and workload, e.g.
`https://runai.example/prod-ai-core-platform/misa-omnivoice-flash-api`.

## One line

```bash
curl -X POST "$BASE/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","input":"Xin chào.","language":"Vietnamese","response_format":"wav"}' \
  -o out.wav
```

Returns the audio bytes with the format's MIME type.

## Clone a voice

Multipart exists for this: attaching a reference with `-F ref_audio=@voice.wav`
beats base64-ing a WAV into a JSON string, and it is how OpenAI's own audio
routes that carry a file are called.

```bash
curl -X POST "$BASE/v1/audio/speech" \
  -F "model=omnivoice" \
  -F "language=Vietnamese" \
  -F "input=This line is read in the cloned voice." \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=exactly what is said in reference.wav" \
  -o cloned.wav
```

`ref_text` is **required** with `ref_audio`, and must be the reference's actual
transcript. Upstream OmniVoice falls back to transcribing it with Whisper; that
is a second model on the serving path and a second way to fail, so it is not
served here. A reference without its text returns `400`.

In JSON, `ref_audio` takes a data URL, a bare base64 blob, a path the *server*
can read, or an `http(s)` URL:

```bash
curl -X POST "$BASE/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","input":"Xin chào.","language":"Vietnamese",
       "ref_audio":"/data/voices/reference.wav",
       "ref_text":"exactly what is said in reference.wav"}' \
  -o cloned.wav
```

## A batch

Pass a list to `input` — repeat the field in multipart. The items are submitted
together so the scheduler can put them in one batch. That is where the
throughput comes from, and why there is no separate batch route.

```bash
curl -X POST "$BASE/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{"model":"omnivoice","language":"Vietnamese","response_format":"wav",
       "input":["Line one.","Line two.","Line three."]}' \
  -o batch.json
```

A list returns JSON rather than audio bytes, one entry per item, in order:

```json
{"object":"list","model":"omnivoice","created":1789449600,"inference_time_s":0.737,
 "data":[{"object":"audio.speech","index":0,"output_format":"wav","b64_json":"..."},
         {"object":"audio.speech","index":1,"output_format":"wav","b64_json":"..."}]}
```

```bash
python3 -c "
import json, base64
for it in json.load(open('batch.json'))['data']:
    open(f\"line_{it['index']}.wav\", 'wb').write(base64.b64decode(it['b64_json']))"
```

## A batch in one voice

One `ref_audio` covers the whole list, and is decoded once rather than once per
line. For a different voice per line, send separate requests.

```bash
curl -X POST "$BASE/v1/audio/speech" \
  -F "model=omnivoice" \
  -F "language=Vietnamese" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=exactly what is said in reference.wav" \
  -F "input=Line one." \
  -F "input=Line two." \
  -o batch.json
```

## Fields

| Field | Default | Meaning |
| --- | --- | --- |
| `input` | — | One string, or a list of them for a batch. Required. |
| `language` | — | Goes into the prompt verbatim, so use the language's name (`Vietnamese`), not a code. Not validated: an unrecognised value reaches the model and skews the voice with nothing in the response to say so. |
| `response_format` | `wav` | `wav` and `pcm` always work. `flac`, `ogg` and `mp3` need the optional `soundfile` backend and **fall back to WAV** without it, so check the content type rather than assume. `pcm` is headerless 16-bit at the model's sample rate. |
| `ref_audio` | — | Reference voice to clone. Multipart file, data URL, bare base64, a path the server can read, or an `http(s)` URL. |
| `ref_text` | — | Transcript of `ref_audio`, and **required** with it. |
| `voice` | — | A *description* of a voice to design, not the name of a preset. There is no preset registry, so `GET /v1/audio/voices` does not exist. |
| `speed` | `1.0` | Playback rate. |
| `seed` | — | Sampling seed. Without it, two identical requests return different audio. |
| `stream` | `false` | Streams a WAV. One string only; streaming a list is refused. |

`temperature` and `top_p` are deliberately not mapped. OmniVoice ranks a whole
canvas by confidence rather than sampling one token per position, so the nearest
knobs are `class_temperature` and `position_temperature`, passed as extra fields.

## Errors

| Status | Cause |
| --- | --- |
| `400` | `ref_audio` without `ref_text`, a list longer than `MSTAR_SPEECH_MAX_BATCH` (16), `stream` with a list, unparseable JSON, or `voice_type`. |
| `422` | The body parsed but a field is missing or the wrong type; the message names the field. |

## Calling it like the TTS wrapper

Clients written against the internal TTS wrapper keep their body and change only
the URL. Its field names are accepted as aliases, and its two routes
(`/inference` and `/batch-inference`) both map onto this one. When both
spellings appear, the OpenAI name wins.

| Wrapper | Here | Note |
| --- | --- | --- |
| `text` | `input` | |
| `texts` | `input` (list) | A JSON array in one field, as the wrapper sends it. |
| `encode_type` | `response_format` | |
| `language=vi` | `language=Vietnamese` | ISO codes are translated to the language's name. |
| `voice_type` | — | **Rejected with 400.** It names a preset in the wrapper's own `ref_voices/` directory, which this server does not have. Aliasing it onto `voice` would return a different voice and no error, so it is refused instead. Clone with `ref_audio` or describe with `voice`. |

```bash
curl -X POST "$BASE/v1/audio/speech" \
  -F 'texts=["Line one.","Line two."]' \
  -F "language=vi" \
  -F "encode_type=pcm" \
  -F "ref_audio=@reference.wav" \
  -F "ref_text=exactly what is said in reference.wav" \
  -o batch.json
```

One difference is deliberate. The wrapper's batch route concatenates every line
into a single audio stream with no boundaries, so the caller has to hunt for
silence to split them — which breaks on the first line containing a comma. A
batch here returns one addressable entry per line, so that code goes away.
