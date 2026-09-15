Using a Server
==============

Once a server is running (see :doc:`serving`), you can reach it three ways: the native
``/generate`` endpoint, the Python SDK, or the OpenAI-compatible API. Every model is
reachable via ``/generate`` and the SDK; the OpenAI routes cover the chat, speech, and
image models.

Native ``/generate``
--------------------

``POST /generate`` takes a multipart form and returns either a single JSON document or an
NDJSON stream.

.. list-table:: Form fields
   :header-rows: 1
   :widths: 22 14 64

   * - Field
     - Default
     - Meaning
   * - ``text``
     - —
     - Text prompt (optional if media is provided).
   * - ``files``
     - —
     - One or more media uploads; each file's modality is inferred from its extension.
   * - ``input_modalities``
     - auto
     - Comma-separated input modalities, one entry per prompt element in order.
       Auto-detected from the uploads and text when omitted, which is what keeps
       the ordering and the count of same-modality attachments; an explicit list
       replaces it.
   * - ``output_modalities``
     - ``text``
     - Comma-separated desired outputs (e.g. ``text``, ``image``, ``audio``, ``video``,
       ``action``).
   * - ``streaming``
     - ``true``
     - ``true`` → NDJSON stream of chunks; ``false`` → one JSON document.
   * - ``model_kwargs``
     - —
     - JSON object of model-specific parameters (e.g. ``{"voice": "tara"}``).
   * - ``request_id``
     - *(uuid)*
     - Optional client-supplied id; the server generates one when omitted.

A non-streaming response groups outputs by modality, each payload base64-encoded:

.. code-block:: json

   {
     "request_id": "…",
     "outputs": {
       "text":  [{"data": "<base64>",     "metadata": {}}],
       "image": [{"data": "<base64-png>", "metadata": {}}]
     }
   }

A streaming response is ``application/x-ndjson`` — one JSON object per line as chunks
arrive. ``GET /health`` returns ``{"status": "healthy"}``.

.. code-block:: bash

   # text (non-streaming → JSON)
   curl -s http://localhost:8000/generate -F 'text=Hello' -F 'streaming=false'

   # image understanding (image in, text out)
   curl -s http://localhost:8000/generate -F 'text=What is in this image?' -F 'files=@cat.jpg'

   # text-to-speech (base64 PCM in outputs.audio)
   curl -s http://localhost:8000/generate \
     -F 'text=hello there' -F 'output_modalities=audio' \
     -F 'model_kwargs={"voice":"tara"}' -F 'streaming=false'

Python SDK
----------

The SDK (:class:`mstar.client.MStarClient`) is a thin HTTP client over ``/generate``. It
depends only on ``requests`` (plus ``numpy`` for the audio helpers) — no torch — so it can
run anywhere:

.. code-block:: python

   from mstar import MStarClient
   client = MStarClient("http://localhost:8000")   # optional: timeout=600.0

The core method is ``generate``:

``generate(*, text=None, images=None, audio=None, video=None, output_modalities=("text",), input_modalities=None, stream=False, request_id=None, **model_kwargs)``
   Submit a request. ``images`` / ``audio`` / ``video`` accept a path, raw ``bytes``, a
   ``(filename, bytes)`` tuple, or a list of those. Extra keyword args are forwarded as the
   model's ``model_kwargs`` (e.g. ``voice="tara"``, ``temperature=0.7``,
   ``max_output_tokens=256``); ``None`` values are dropped. Returns a ``GenerateResult``
   when ``stream=False``, or an iterator of stream events when ``stream=True``.

Convenience wrappers:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Returns
   * - ``chat(prompt, *, images=None, audio=None, output_modalities=("text",), stream=False, **kw)``
     - Text generation (and, with ``output_modalities=("text", "audio")``, speech).
   * - ``generate_image(prompt, **kw)``
     - PNG ``bytes`` (e.g. BAGEL text-to-image).
   * - ``tts(text, *, voice=None, **kw)``
     - An ``AudioBuffer`` (``.to_wav(path)``, ``.to_numpy()``, ``len(...)`` samples).
   * - ``stream(**kw)``
     - Sugar for ``generate(stream=True, ...)``.
   * - ``health()``
     - ``True`` if the server is healthy.

Result and event types live in ``mstar.client``:

- ``GenerateResult`` — ``.text``, ``.images`` (list of PNG bytes), ``.audio``
  (an ``AudioBuffer`` or ``None``), ``.raw``; plus ``.save_image(path)`` /
  ``.save_audio(path)``.
- ``AudioBuffer`` — decoded PCM with ``.sample_rate``; ``.to_wav(path)``, ``.to_numpy()``,
  ``len(...)``.
- Stream events — ``TextChunk(text)``, ``ImageChunk(data)`` (``.save(path)``),
  ``AudioChunk(pcm, sample_rate)``.

.. code-block:: python

   res = client.chat("Hello!")                       # GenerateResult
   print(res.text)

   open("cat.png", "wb").write(client.generate_image("a cat in a hat"))

   client.tts("Hi there", voice="tara").to_wav("out.wav")

   for event in client.stream(text="Tell me a story"):
       print(getattr(event, "text", ""), end="", flush=True)

OpenAI-compatible API
---------------------

``mstar`` mounts OpenAI-style routes under ``/v1`` for the models with standard OpenAI
semantics. Point any OpenAI client at ``http://<host>:<port>/v1``:

.. code-block:: python

   from openai import OpenAI
   client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

Endpoints and model coverage:

.. list-table::
   :header-rows: 1
   :widths: 34 22 44

   * - Endpoint
     - Models
     - Notes
   * - ``GET /v1/models``
     - all
     - Lists the served model.
   * - ``POST /v1/chat/completions``
     - ``bagel``, ``qwen3_omni``
     - Text chat (streaming + non-streaming). Qwen3-Omni can also emit speech.
   * - ``POST /v1/audio/speech``
     - ``orpheus``, ``qwen3_omni``, ``omnivoice``
     - Text-to-speech. OmniVoice also clones a voice and takes a batch; see
       :ref:`omnivoice-speech`.
   * - ``POST /v1/images/generations``
     - ``bagel``
     - Text-to-image.
   * - ``POST /v1/images/edits``
     - ``bagel``
     - Image editing (image + prompt → image).

Models without an OpenAI surface (``pi05``, ``vjepa2``, ``vjepa2_ac``) return ``404`` on
``/v1/*``; use ``/generate`` or the SDK for them.

.. code-block:: python

   # chat
   client.chat.completions.create(model="bagel", messages=[{"role": "user", "content": "hi"}])

   # text-to-speech
   client.audio.speech.create(model="orpheus", input="hello there", voice="tara")

   # image generation
   client.images.generate(model="bagel", prompt="a cat in a hat")

Per-model notes:

- **BAGEL** — chat returns text only; use ``/v1/images/generations`` and
  ``/v1/images/edits`` for image output.
- **Qwen3-Omni** — text sampling uses ``thinker_*`` keys, speech uses ``talker_*``, and the
  residual codec groups use ``code_predictor_*``; set the speaker with ``voice`` (default
  ``Ethan``) and request audio output by including ``"audio"`` in ``modalities``.
  Non-OpenAI knobs (e.g. ``talker_top_k``, ``code_predictor_top_p``) go through
  ``extra_body``.
- **OmniVoice** — clone a voice with ``ref_audio``, or design one by describing it in
  ``voice``; synthesise several lines in one call by passing a list to ``input``. Full
  reference below. ``temperature`` and ``top_p`` are deliberately not mapped: OmniVoice
  ranks a whole canvas by confidence rather than sampling one token per position, so the
  nearest knobs are ``class_temperature`` and ``position_temperature``, passed through
  ``extra_body``.
- **Orpheus** — set the speaker with ``voice`` — one of ``tara`` (default), ``zoe``,
  ``zac``, ``jess``, ``leo``, ``mia``, ``julia``, ``leah`` (the ``available_voices`` list
  in the Orpheus config).


.. _omnivoice-speech:

OmniVoice speech
----------------

Every example below is a complete ``curl``. Replace ``$BASE`` with the server root --
on Run:AI that includes the project and workload, e.g.
``https://runai.example/prod-ai-core-platform/misa-omnivoice-flash-api``.

The endpoint takes **either JSON or multipart**. Multipart exists because this route
carries a file: attaching a reference voice with ``-F ref_audio=@voice.wav`` beats
base64-ing a WAV into a JSON string, and it is how OpenAI's own audio routes that take a
file are called.

One line
~~~~~~~~

.. code-block:: bash

   curl -X POST "$BASE/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{"model":"omnivoice","input":"Xin chào.","language":"Vietnamese","response_format":"wav"}' \
     -o out.wav

Returns the audio bytes with the format's MIME type.

Clone a voice
~~~~~~~~~~~~~

.. code-block:: bash

   curl -X POST "$BASE/v1/audio/speech" \
     -F "model=omnivoice" \
     -F "language=Vietnamese" \
     -F "input=This line is read in the cloned voice." \
     -F "ref_audio=@reference.wav" \
     -o cloned.wav

``ref_text`` is optional -- without it the reference is transcribed with Whisper. Supply
it when you already have the transcript and want to skip that step, or when the automatic
transcript comes out wrong.

In JSON, ``ref_audio`` takes a data URL, a bare base64 blob, a path the *server* can read,
or an ``http(s)`` URL:

.. code-block:: bash

   curl -X POST "$BASE/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{"model":"omnivoice","input":"Xin chào.","language":"Vietnamese",
          "ref_audio":"/data/voices/reference.wav"}' \
     -o cloned.wav

A batch
~~~~~~~

Pass a list to ``input`` (repeat the field in multipart). The items are submitted together
so the scheduler can put them in one batch -- that is where the throughput comes from, and
why there is no separate batch route.

.. code-block:: bash

   curl -X POST "$BASE/v1/audio/speech" \
     -H "Content-Type: application/json" \
     -d '{"model":"omnivoice","language":"Vietnamese","response_format":"wav",
          "input":["Line one.","Line two.","Line three."]}' \
     -o batch.json

A list returns JSON rather than audio bytes, one entry per item, in order:

.. code-block:: json

   {"object":"list","model":"omnivoice","created":1789449600,"inference_time_s":2.1,
    "data":[{"object":"audio.speech","index":0,"output_format":"wav","b64_json":"..."},
            {"object":"audio.speech","index":1,"output_format":"wav","b64_json":"..."}]}

.. code-block:: bash

   python3 -c "
   import json, base64
   for it in json.load(open('batch.json'))['data']:
       open(f\"line_{it['index']}.wav\", 'wb').write(base64.b64decode(it['b64_json']))"

A batch in one voice
~~~~~~~~~~~~~~~~~~~~

One ``ref_audio`` covers the whole list; it is decoded once, not once per line.

.. code-block:: bash

   curl -X POST "$BASE/v1/audio/speech" \
     -F "model=omnivoice" \
     -F "language=Vietnamese" \
     -F "ref_audio=@reference.wav" \
     -F "input=Line one." \
     -F "input=Line two." \
     -o batch.json

For a different voice per line, send separate requests.

Fields
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 16 60

   * - Field
     - Default
     - Meaning
   * - ``input``
     - —
     - One string, or a list of them for a batch. Required.
   * - ``language``
     - —
     - Goes into the prompt verbatim, so use the language's name
       (``Vietnamese``), not a code. Not validated: an unrecognised value is
       passed to the model and skews the voice with nothing in the response to
       say so.
   * - ``response_format``
     - ``wav``
     - ``wav`` and ``pcm`` are always available. ``flac``, ``ogg`` and ``mp3``
       need the optional ``soundfile`` backend, and **fall back to WAV** if it
       is missing -- check the response's content type rather than assuming.
       ``pcm`` is headerless 16-bit at the model's sample rate.
   * - ``ref_audio``
     - —
     - Reference voice to clone. Multipart file, data URL, bare base64, a path
       the server can read, or an ``http(s)`` URL.
   * - ``ref_text``
     - —
     - Transcript of ``ref_audio``. Optional; Whisper transcribes it otherwise.
   * - ``voice``
     - —
     - A *description* of a voice to design, not the name of a preset. There is
       no preset registry, so ``GET /v1/audio/voices`` does not exist.
   * - ``speed``
     - ``1.0``
     - Playback rate.
   * - ``seed``
     - —
     - Sampling seed. Without it, two identical requests return different audio.
   * - ``stream``
     - ``false``
     - Streams a WAV. One string only; streaming a list is refused.

Errors
~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Status
     - Cause
   * - ``400``
     - A list longer than ``MSTAR_SPEECH_MAX_BATCH`` (16), ``stream`` with a
       list, unparseable JSON, or ``voice_type`` (see below).
   * - ``422``
     - The body parsed but a field is missing or the wrong type; the message
       names the field.

Calling it like the in-house TTS wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clients written against the internal TTS wrapper can keep their body and change only the
URL. Its field names are accepted as aliases, and its two routes (``/inference`` and
``/batch-inference``) both map onto this one. When both spellings appear, the OpenAI name
wins.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Wrapper
     - Here
     - Note
   * - ``text``
     - ``input``
     -
   * - ``texts``
     - ``input`` (list)
     - A JSON array in one field, as the wrapper sends it.
   * - ``encode_type``
     - ``response_format``
     -
   * - ``language=vi``
     - ``language=Vietnamese``
     - ISO codes are translated to the language's name.
   * - ``voice_type``
     - —
     - **Rejected with 400.** It names a preset in the wrapper's own
       ``ref_voices/`` directory, which this server does not have. Aliasing it
       onto ``voice`` would return a different voice and no error, so it is
       refused instead. Clone with ``ref_audio`` or describe with ``voice``.

.. code-block:: bash

   curl -X POST "$BASE/v1/audio/speech" \
     -F 'texts=["Line one.","Line two."]' \
     -F "language=vi" \
     -F "encode_type=pcm" \
     -F "ref_audio=@reference.wav" \
     -o batch.json

One difference is deliberate. The wrapper's batch route concatenates every line into a
single audio stream with no boundaries, so the caller has to hunt for silence to split
them -- which breaks on the first line containing a comma. A batch here returns one
addressable entry per line, so that code goes away.

