# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Text embedding against the configured gpuTextToVec endpoint.

The endpoint takes a JSON body ``{"texts": [...], "model_name": "..."}`` and
authenticates with an ``x-jwt-ms-token`` header, so it does not speak the
OpenAI ``/embeddings`` API and is called directly over httpx rather than via
the OpenAI client.

Ingest and query MUST use the same model; a mismatch produces garbage results
or a dimension error from pgvector.
"""

from __future__ import annotations

import itertools
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator, Literal

import httpx

from auto_ontology.utils.model_config import resolve
from auto_ontology.vdb.records import write_embedded_rows

if TYPE_CHECKING:
    import pandas as pd

    from auto_ontology.vdb.postgres import PostgresVDB

logger = logging.getLogger(__name__)

InputType = Literal["passage", "query"]

# The gpuTextToVec endpoint authenticates with a JWT carried in this header
# rather than an ``Authorization: Bearer`` scheme, so the key configured as
# EMBED_API_KEY is sent here, not as a bearer token.
_AUTH_HEADER = "x-jwt-ms-token"

# HTTP statuses that are transient: the per-request retry loop below retries
# these before surfacing the failure. Anything else >= 400 is a wrong request
# (bad key, unknown model, wrong path) and is raised at once as fatal.
_TRANSIENT_STATUSES = {408, 409, 429, 500, 502, 503, 504}

# Per-request retry backoff: short, because the slow rate-limit recovery is the
# chunk-level retry's job (see _embed_with_retry). Seconds, doubling, capped.
_RETRY_BACKOFF_BASE_S = 0.25
_RETRY_BACKOFF_MAX_S = 4.0

# Each field falls back to DEFAULT_MODELS_<field>.
_EMBED_ENDPOINT = resolve("EMBED", "ENDPOINT")
_EMBED_MODEL = resolve("EMBED", "MODEL")
_EMBED_API_KEY = resolve("EMBED", "API_KEY")

# Per-request retries inside the client, for 408/409/429/5xx and dropped
# connections. They back off for seconds; the chunk-level retry below is the
# slow one, for a rate limit that needs minutes to clear.
_REQUEST_RETRIES = 5

# Interactive writes (an edited column, a saved analysis) run inside a
# request handler, so a dead endpoint has to surface in about a minute: one
# 60s attempt and one retry for a transient blip, instead of the ingest-sized
# 600s and five retries.
_INTERACTIVE_TIMEOUT_S = 60.0
_INTERACTIVE_RETRIES = 1

# Retry policy for a chunk the endpoint refused. Delays are long on purpose:
# the failure being retried is a rate limit, and the quota needs wall-clock time
# to refill — an immediate retry just burns another attempt against a closed
# window. Measured recovery after saturating the endpoint was under 3 minutes,
# which 15s doubling to 120s covers across four attempts.
_RETRY_ATTEMPTS = 4
_RETRY_BASE_SECONDS = 15.0
_RETRY_MAX_SECONDS = 120.0

# Why a chunk came back unembedded, kept on the frame rather than on every row.
_ERROR_ATTR = "embed_error"
# Set when retrying cannot help: the request is wrong (bad key, unknown model,
# wrong path), not the endpoint busy.
_FATAL_ATTR = "embed_error_fatal"


@dataclass(frozen=True)
class EmbedParams:
    """Where and how to embed. The key is kept out of ``repr`` and so out of logs."""

    model_name: str
    embed_invoke_url: str
    api_key: str = field(default="", repr=False)
    embed_modality: str = "text"
    inference_batch_size: int = 32
    nim_http_max_concurrent: int = 32
    request_timeout_s: float = 600.0


def get_embed_params() -> EmbedParams:
    return EmbedParams(
        embed_invoke_url=_EMBED_ENDPOINT,
        model_name=_EMBED_MODEL,
        api_key=_EMBED_API_KEY,
    )


class TransientEmbeddingError(RuntimeError):
    """A transient embedding-endpoint failure (5xx/429/timeout/transport).

    Raised by :func:`_post_embeddings` when its own retry budget is exhausted
    on a failure waiting cannot rule out, so callers can tell a blip from a
    wrong request (the latter raises :class:`httpx.HTTPStatusError` for a 4xx)
    by type rather than by message text.
    """


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

# One httpx.Client per endpoint URL, so its connection pool is shared across
# batches and across embed_texts calls.
_http_clients: dict[str, httpx.Client] = {}
_http_clients_lock = threading.Lock()


def _http_client(params: EmbedParams) -> httpx.Client:
    """A shared httpx.Client keyed by endpoint URL, so connection pools are reused."""
    url = params.embed_invoke_url
    with _http_clients_lock:
        client = _http_clients.get(url)
        if client is None:
            client = _http_clients[url] = httpx.Client()
        return client


def _retry_backoff(attempt: int) -> float:
    return min(_RETRY_BACKOFF_BASE_S * (2**attempt), _RETRY_BACKOFF_MAX_S)


def _extract_vectors(payload: object, expected: int) -> list[list[float]]:
    """Pull the vectors out of the endpoint's JSON response, in input order.

    The gpuTextToVec response envelope is not documented here, so the common
    shapes are all accepted: a top-level list, or an object under
    ``embeddings``, ``vectors``, ``result`` or ``data``. ``data`` may itself be
    a list of vectors or a list of ``{"embedding": [...]}`` objects (the OpenAI
    shape, kept so a NIM-style endpoint still works if it is wired in later).

    The count must match *expected*: one vector per input text, in order. A
    short or long response is a loud error rather than silently dropping or
    duplicating rows.
    """
    if isinstance(payload, list):
        raw = payload
    elif isinstance(payload, dict):
        for key in ("embeddings", "vectors", "result", "data"):
            value = payload.get(key)
            if value is not None:
                raw = value
                break
        else:
            raise RuntimeError(
                "embedding endpoint returned an unrecognised body: "
                f"{str(payload)[:200]}"
            )
    else:
        raise RuntimeError(
            f"embedding endpoint returned a non-object body: {str(payload)[:200]}"
        )
    vectors: list[list[float]] = []
    for item in raw:
        if isinstance(item, dict) and "embedding" in item:
            vectors.append([float(x) for x in item["embedding"]])
        elif isinstance(item, (list, tuple)):
            vectors.append([float(x) for x in item])
        else:
            raise RuntimeError(
                f"embedding endpoint returned a non-vector element: {item!r}"
            )
    if len(vectors) != expected:
        raise RuntimeError(
            f"embedding endpoint returned {len(vectors)} vector(s) "
            f"for {expected} input(s)."
        )
    return vectors


def _post_embeddings(
    params: EmbedParams,
    texts: list[str],
    *,
    timeout: float,
    retries: int,
) -> list[list[float]]:
    """POST *texts* to the gpuTextToVec endpoint and return one vector each.

    Retries transient failures (408/409/429/5xx and transport errors) up to
    *retries* times with a short backoff. A non-transient 4xx is raised at once
    via ``raise_for_status`` so :func:`_is_fatal` can classify it by status.
    """
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if params.api_key:
        headers[_AUTH_HEADER] = params.api_key
    payload = {"texts": list(texts), "model_name": params.model_name}
    client = _http_client(params)
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = client.post(
                params.embed_invoke_url,
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(_retry_backoff(attempt))
                continue
            break
        if response.status_code in _TRANSIENT_STATUSES:
            last_exc = RuntimeError(
                f"embedding endpoint returned HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )
            if attempt < retries:
                time.sleep(_retry_backoff(attempt))
                continue
            break
        if response.status_code >= 400:
            response.raise_for_status()
        return _extract_vectors(response.json(), len(texts))
    raise TransientEmbeddingError(
        f"embedding request to {params.embed_invoke_url} failed after "
        f"{retries + 1} attempt(s): {last_exc}"
    )


def embed_texts(
    texts: list[str],
    params: EmbedParams,
    input_type: InputType,
    *,
    timeout: float | None = None,
    retries: int | None = None,
) -> list[list[float] | None]:
    """Embed *texts*, returning one vector per text in the same order.

    Blank texts are not sent and come back as ``None``. Raises the client's
    error when a request fails after its retries -- an
    :class:`httpx.HTTPStatusError` carries the HTTP status in its message.
    *timeout* (seconds per request) and *retries* (per request) default to the
    ingest-sized ``params.request_timeout_s`` and ``_REQUEST_RETRIES``; a caller
    with its own retry loop passes ``retries=0`` so the two do not multiply.
    """
    stripped = [text.strip() if isinstance(text, str) else "" for text in texts]
    to_send = [i for i, text in enumerate(stripped) if text]
    vectors: list[list[float] | None] = [None] * len(texts)
    if not to_send:
        return vectors

    max_retries = _REQUEST_RETRIES if retries is None else retries
    batches = list(itertools.batched(to_send, max(1, params.inference_batch_size)))
    request_timeout = timeout or params.request_timeout_s

    def _embed_batch(indices: tuple[int, ...]) -> None:
        batch_texts = [stripped[i] for i in indices]
        batch_vectors = _post_embeddings(
            params,
            batch_texts,
            timeout=request_timeout,
            retries=max_retries,
        )
        for i, vector in zip(indices, batch_vectors):
            vectors[i] = list(vector)

    if len(batches) == 1:
        _embed_batch(batches[0])
    else:
        workers = min(len(batches), max(1, params.nim_http_max_concurrent))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # list() so the first failed batch re-raises here.
            list(pool.map(_embed_batch, batches))
    return vectors


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


def _http_status(exc: Exception) -> int | None:
    """The HTTP status carried by *exc*, if any.

    ``httpx.HTTPStatusError`` exposes it on ``.response.status_code``; matched
    by attribute rather than by import so a driver swap degrades to "not fatal"
    instead of raising from inside the error path.
    """
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None)


def _is_fatal(exc: Exception) -> bool:
    """Whether *exc* says the request itself is wrong, so waiting cannot help.

    Any 4xx except 408 (timeout), 409 and 429 (rate limit): a bad key, an
    unknown model id, a wrong path. Transport errors and exhausted transient
    retries are not fatal -- the chunk-level retry may still recover them.
    """
    status = _http_status(exc)
    if status is None:
        return False
    return 400 <= status < 500 and status not in (408, 409, 429)


def _embed_frame(frame: "pd.DataFrame", params: EmbedParams) -> "pd.DataFrame":
    """Embed *frame*'s ``text`` column as passages into ``metadata["embedding"]``.

    Never raises. A failure leaves every row without an embedding and the
    reason in ``frame.attrs``, which is what lets the caller retry the chunk,
    log why it failed, and still write the chunks that worked.
    """
    out = frame.copy()
    # A fresh dict per row: the caller's row dicts are not ours to mutate.
    out["metadata"] = [
        dict(m) if isinstance(m, dict) else {}
        for m in out.get("metadata", [None] * len(out))
    ]
    try:
        vectors = embed_texts(list(out.get("text", [""] * len(out))), params, "passage")
    except Exception as exc:
        out.attrs[_ERROR_ATTR] = f"{type(exc).__name__}: {exc}"
        out.attrs[_FATAL_ATTR] = _is_fatal(exc)
        return out
    for metadata, vector in zip(out["metadata"], vectors):
        if vector is not None:
            metadata["embedding"] = vector
    return out


def embed_chunk_size(params: EmbedParams) -> int:
    """Rows per embed call.

    One chunk is one full wave through the endpoint — ``batch_size`` texts per
    request times ``max_concurrent`` requests in flight — so the HTTP pool stays
    saturated while each chunk still finishes in seconds rather than minutes.
    That cadence is what the per-chunk log line, the streaming write and the
    retry all hang off: smaller chunks mean more frequent progress and less work
    lost to — or redone by — a mid-run failure, larger ones mean less pgvector
    round-tripping.
    """
    return max(1, params.inference_batch_size * params.nim_http_max_concurrent)


def _embedded_row_count(frame: "pd.DataFrame") -> int:
    """Rows in *frame* that came back with a non-empty embedding."""
    return sum(
        1
        for metadata in frame.get("metadata", [])
        if isinstance(metadata, dict) and metadata.get("embedding")
    )


def _failure_reason(frame: "pd.DataFrame | None") -> str:
    """The endpoint's error for a chunk that came back unembedded."""
    if frame is None or frame.empty:
        return "embed returned no rows"
    return (
        frame.attrs.get(_ERROR_ATTR) or "no embeddings returned and no error reported"
    )


def _embed_with_retry(
    embed: "Callable[[pd.DataFrame], pd.DataFrame | None]",
    part: "pd.DataFrame",
    *,
    label: str,
    attempts: int = _RETRY_ATTEMPTS,
    sleep: "Callable[[float], None]" = time.sleep,
) -> "pd.DataFrame | None":
    """Embed *part*, retrying while the endpoint returns nothing usable.

    Retries the whole chunk on top of the client's per-request retries because
    a sustained run exhausts the endpoint's quota for minutes, longer than those
    back off for, and without this the chunk's rows are silently dropped and the
    catalog ends up missing entries no one notices. A request the endpoint
    rejects as wrong -- a bad key, an unknown model -- is not retried: waiting
    does not fix it.

    Returns the embedded frame, or the last empty result once attempts run out;
    the caller decides what to do with a chunk that never succeeded.
    """
    delay = _RETRY_BASE_SECONDS
    embedded = None
    for attempt in range(1, attempts + 1):
        embedded = embed(part)
        if (
            embedded is not None
            and not embedded.empty
            and _embedded_row_count(embedded)
        ):
            if attempt > 1:
                logger.info("%s: recovered on attempt %d/%d", label, attempt, attempts)
            return embedded
        reason = _failure_reason(embedded)
        fatal = embedded is not None and embedded.attrs.get(_FATAL_ATTR, False)
        if attempt == attempts or fatal:
            logger.error(
                "%s: giving up after %d attempt(s) — %s", label, attempt, reason
            )
            return embedded
        logger.warning(
            "%s: attempt %d/%d failed (%s); retrying in %.0fs",
            label,
            attempt,
            attempts,
            reason,
            delay,
        )
        sleep(delay)
        delay = min(delay * 2, _RETRY_MAX_SECONDS)
    return embedded


def batch_embed_chunks(
    rows: "list[dict] | pd.DataFrame",
    params: EmbedParams,
    *,
    label: str = "",
    chunk_size: int | None = None,
) -> "Iterator[pd.DataFrame]":
    """Embed *rows* chunk by chunk, yielding each chunk as it comes back.

    Generating instead of returning is what lets the caller write each chunk to
    the VDB while the next one is still embedding — see
    :func:`auto_ontology.ingestion_service.ingest.run_ingest`. Yields nothing when *rows*
    is empty; the caller decides whether that is an error.
    """
    import pandas as pd

    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    total = len(frame)
    if total == 0:
        return

    size = chunk_size or embed_chunk_size(params)
    chunks = math.ceil(total / size)
    prefix = f"Embed{f' {label}' if label else ''}"
    logger.info(
        "%s: %d row(s) in %d chunk(s) of %d via %s (model %s, batch %s, "
        "concurrency %s)",
        prefix,
        total,
        chunks,
        size,
        params.embed_invoke_url,
        params.model_name,
        params.inference_batch_size,
        params.nim_http_max_concurrent,
    )

    started = time.monotonic()
    done = 0
    embedded_total = 0

    for index in range(chunks):
        part = frame.iloc[index * size : (index + 1) * size]
        chunk_label = f"{prefix} chunk {index + 1}/{chunks}"
        chunk_started = time.monotonic()
        embedded = _embed_with_retry(
            lambda p: _embed_frame(p, params), part, label=chunk_label
        )
        chunk_elapsed = time.monotonic() - chunk_started

        if embedded is None or embedded.empty:
            logger.warning(
                "%s: embed returned nothing for %d row(s) in %.1fs",
                chunk_label,
                len(part),
                chunk_elapsed,
            )
            continue

        with_embeddings = _embedded_row_count(embedded)
        done += len(part)
        embedded_total += with_embeddings
        elapsed = time.monotonic() - started
        rate = done / elapsed if elapsed > 0 else 0.0
        log = logger.info if with_embeddings else logger.warning
        log(
            "%s chunk %d/%d: %d/%d row(s) embedded in %.1fs "
            "(%d/%d done, %.0f rows/s, ~%.0fs left)",
            prefix,
            index + 1,
            chunks,
            with_embeddings,
            len(part),
            chunk_elapsed,
            done,
            total,
            rate,
            (total - done) / rate if rate > 0 else 0.0,
        )
        yield embedded

    logger.info(
        "%s: %d/%d row(s) embedded in %.1fs",
        prefix,
        embedded_total,
        total,
        time.monotonic() - started,
    )


def batch_embed(
    rows: "list[dict] | pd.DataFrame",
    params: EmbedParams,
) -> "pd.DataFrame":
    """Embed *rows* and return one DataFrame with the embeddings.

    A concatenating wrapper over :func:`batch_embed_chunks`, for the callers
    that want the whole result in hand. Prefer the generator when the rows are
    going straight into a VDB — it bounds peak memory and starts persisting
    before the last chunk is embedded.

    Returns an empty DataFrame when *rows* is empty or the embed step yields
    nothing; the caller decides whether that is an error.
    """
    import pandas as pd

    frames = list(batch_embed_chunks(rows, params))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def embed_rows(
    rows: list[dict[str, Any]],
    params: EmbedParams,
    *,
    what: str,
) -> list[dict[str, Any]]:
    """Embed already-built rows as passages, all or nothing.

    For the small, interactive writes -- one analysis, one attribute, one
    edited column -- where a failure should reach the caller instead of being
    retried for minutes the way an ingest chunk is: one 60s attempt and one
    retry, so a dead endpoint fails the request in about two minutes rather
    than an hour. Returns copies of the rows
    that came back with an embedding, ready for
    :func:`~auto_ontology.vdb.records.write_embedded_rows`. Raises
    ``RuntimeError`` when the endpoint fails or returns no embeddings; *what*
    names the rows in that message.
    """
    if not rows:
        return []
    try:
        vectors = embed_texts(
            [row.get("text") or "" for row in rows],
            params,
            "passage",
            timeout=_INTERACTIVE_TIMEOUT_S,
            retries=_INTERACTIVE_RETRIES,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Embedding {len(rows)} {what} row(s) via {params.embed_invoke_url} "
            f"failed: {exc}"
        ) from exc

    embedded = [
        {**row, "metadata": {**(row.get("metadata") or {}), "embedding": vector}}
        for row, vector in zip(rows, vectors)
        if vector
    ]
    if not embedded:
        raise RuntimeError(
            f"Embedding step produced 0/{len(rows)} {what} rows with embeddings."
        )
    return embedded


def embed_rows_into_vdb(
    rows: list[dict[str, Any]],
    params: EmbedParams,
    vdb: "PostgresVDB",
    *,
    what: str,
) -> int:
    """:func:`embed_rows`, then append the result to *vdb*.

    Returns the number of rows written.
    """
    if not rows:
        return 0
    before = time.time()
    embedded = embed_rows(rows, params, what=what)
    written = write_embedded_rows(vdb, embedded)
    logger.info(
        "Embedded and appended %d/%d %s row(s) via %s in %.2fs.",
        len(embedded),
        len(rows),
        what,
        type(vdb).__name__,
        time.time() - before,
    )
    return written


def embed_docs_into_vdb(
    docs: list[dict],
    embed_params: EmbedParams,
    vdb: "PostgresVDB",
    database_name: str | None = None,
) -> int:
    """Embed *docs* and upsert them into *vdb*.

    Each doc must have at least ``id``, ``name``, ``label``, and ``text`` keys
    (the shape returned by ``fetch_sql_attribute_docs`` and
    ``fetch_suggested_sql_attribute_docs``). An optional ``source`` key is
    carried through to the metadata unchanged (empty string when absent) —
    used by SqlAttribute docs so retrieval hits know their provenance
    (``"sql"``, ``"bridgeTable"``, …) without an extra lookup.

    Returns the number of rows successfully embedded and ingested.
    Raises ``RuntimeError`` when the embedding call produces zero embedded rows
    so the caller can decide how to handle the failure.
    """
    if not docs:
        return 0

    rows = []
    for item in docs:
        node_id = item.get("id")
        path = (
            f"auto_ontology:{node_id}"
            if node_id is not None
            else "auto_ontology:unknown"
        )
        tabular_fields = {
            "id": node_id,
            "label": item.get("label", ""),
            "name": item.get("name", ""),
            "source_path": path,
            "database_name": database_name,
            "source": item.get("source", ""),
        }
        rows.append(
            {
                "text": (item.get("text") or "").strip(),
                "path": path,
                "page_number": -1,
                "metadata": {
                    **tabular_fields,
                    "content_metadata": dict(tabular_fields),
                },
            }
        )
    label = str(docs[0].get("label") or "document")
    return embed_rows_into_vdb(rows, embed_params, vdb, what=label)
