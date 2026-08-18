"""HTTP client for a local Ollama server.

PAPER TRADING ONLY — this client talks exclusively to the local Ollama
inference server (``settings.ollama_url``). It never contacts an exchange and
never handles API keys or secrets.

The client is deliberately small: list installed models, check availability,
and run a single non-streaming chat completion with retries, model fallback
and cleanup of ``<think>...</think>`` reasoning blocks emitted by models such
as qwen3 and deepseek-r1.
"""

from __future__ import annotations

import logging
import re
import time

import requests

from config.settings import settings

logger = logging.getLogger(__name__)

# Well-formed <think>...</think> blocks (qwen3 / deepseek-r1 chain-of-thought).
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"^\s*<think>", re.IGNORECASE)

# Short timeouts for the cheap availability/tag endpoints (connect, read).
_TAGS_TIMEOUT: tuple[float, float] = (5.0, 15.0)

# R1-style reasoning models must NOT be constrained with format="json": the
# grammar traps the completion inside their chain-of-thought (observed with
# Fin-R1:Q5 — a lone {"think": ...} object and no verdict). Without the
# constraint they think in <think>...</think> and then emit clean JSON, which
# _strip_think + _extract_json_object already reduce to a bare object.
_NO_JSON_FORMAT_RE = re.compile(r"fin-r1|deepseek-r1", re.IGNORECASE)


class OllamaError(Exception):
    """Raised when the Ollama server cannot produce a usable response."""


def _strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from a model response.

    Handles three cases:
      1. Well-formed blocks — removed with a regex substitution.
      2. A stray/unmatched ``</think>`` — everything up to and including the
         *last* close tag is discarded (the answer follows the reasoning).
      3. An opening ``<think>`` that is never closed (response truncated) —
         best effort: keep everything from the first ``{`` onwards so a JSON
         payload embedded in the reasoning still survives.

    Args:
        text: Raw model output.

    Returns:
        The output with reasoning blocks removed, whitespace-trimmed.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    close_matches = list(_THINK_CLOSE_RE.finditer(cleaned))
    if close_matches:
        # Unmatched close tag(s): keep only what follows the last one.
        cleaned = cleaned[close_matches[-1].end():]
    elif _THINK_OPEN_RE.match(cleaned):
        # Unclosed think block: salvage a JSON object if one is present.
        brace = cleaned.find("{")
        cleaned = cleaned[brace:] if brace != -1 else ""
    return cleaned.strip()


def _extract_json_object(text: str) -> str:
    """Extract the outermost JSON object from text with surrounding prose.

    Returns the substring from the first ``{`` to the last ``}`` when both
    exist; otherwise returns the input unchanged (letting the caller's JSON
    parser surface a proper error).

    Args:
        text: Cleaned model output that should contain a JSON object.

    Returns:
        The best-effort JSON object substring.
    """
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start:end + 1]
    return stripped


class OllamaClient:
    """Thin wrapper over the local Ollama REST API.

    Attributes:
        base_url: Base URL of the Ollama server (no trailing slash).
        timeout: Per-request timeout for chat completions, in seconds.
        retries: Number of retries (after the first attempt) for the primary
            model before falling back.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
    ) -> None:
        """Initialise the client, defaulting every knob from ``settings``.

        Args:
            base_url: Ollama server URL; default ``settings.ollama_url``.
            timeout: Chat request timeout in seconds; default
                ``settings.ollama_timeout``.
            retries: Retry count for the primary model; default
                ``settings.ollama_retries``.
        """
        self.base_url: str = (base_url or settings.ollama_url).rstrip("/")
        self.timeout: float = float(timeout if timeout is not None else settings.ollama_timeout)
        self.retries: int = int(retries if retries is not None else settings.ollama_retries)
        #: Name of the model that produced the last successful ``chat`` reply
        #: (the resolved installed tag, or the fallback when it answered).
        self.last_model_used: str | None = None

    def list_models(self) -> list[str]:
        """Return the names of locally installed models.

        Never raises: any failure (server down, bad payload, timeout) is
        logged and an empty list is returned.

        Returns:
            Model names as reported by ``GET /api/tags``, e.g.
            ``["qwen3:14b", "deepseek-r1:32b"]``; ``[]`` on any failure.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=_TAGS_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            models = payload.get("models") or []
            return [
                str(m["name"])
                for m in models
                if isinstance(m, dict) and m.get("name")
            ]
        except Exception:  # noqa: BLE001 - contract: never raises
            logger.warning("Could not list Ollama models at %s", self.base_url, exc_info=True)
            return []

    @staticmethod
    def _resolve_installed(wanted: str, installed: list[str]) -> str | None:
        """Resolve a requested model name to a concretely installed tag.

        Ollama normalises model names to lowercase and users routinely pull a
        different quantisation tag than the configured one, so a naive exact
        string check would reject a genuinely installed model (and silently
        divert the call to the fallback model). Matching mirrors the
        second-judge availability rule: exact name first, then
        case-insensitive exact, then the base name before the ``:`` tag.

        Args:
            wanted: The requested model name (possibly with a ``:tag``).
            installed: Names from ``list_models()``; an empty list means the
                catalogue is UNKNOWN (tags call failed or no models), in
                which case the requested name is returned unchanged so the
                server itself gets to accept or reject it.

        Returns:
            The installed name to query, or None when the catalogue is known
            and contains no match.
        """
        if not installed:
            return wanted
        if wanted in installed:
            return wanted
        wanted_lower = wanted.strip().lower()
        wanted_base = wanted_lower.split(":", 1)[0]
        base_match: str | None = None
        for name in installed:
            have = str(name).strip().lower()
            if have == wanted_lower:
                return name
            if base_match is None and have.split(":", 1)[0] == wanted_base:
                base_match = name
        return base_match

    def is_available(self) -> bool:
        """Check whether the Ollama server is reachable.

        Returns:
            True when ``GET /api/tags`` answers with a 2xx status.
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=_TAGS_TIMEOUT)
            return bool(resp.ok)
        except requests.RequestException:
            return False

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        json_mode: bool = True,
        temperature: float = 0.2,
        allow_fallback: bool = True,
    ) -> str:
        """Run a single non-streaming chat completion.

        The primary model (``model`` or ``settings.ollama_model``) is first
        resolved against ``list_models()`` — exact match, then
        case-insensitive, then by base name before the ``:`` tag — and the
        concrete installed tag is what gets queried, so a differently-cased
        or differently-quantised install of the requested model is honoured
        instead of being silently diverted to the fallback. The resolved
        model is tried with ``self.retries`` retries and exponential backoff
        (1s, 2s, 4s). If it exhausts its retries — or is absent from a known
        installed list — ``settings.ollama_fallback_model`` is tried once,
        unless ``allow_fallback`` is False, in which case ``OllamaError`` is
        raised instead (for callers that must never accept a substitute
        model's opinion, e.g. the auto-trader's second judge).
        ``<think>...</think>`` reasoning blocks are always stripped; in JSON
        mode the outermost JSON object is extracted from any residual prose.

        Args:
            prompt: User message content.
            system: Optional system prompt.
            model: Model name; defaults to ``settings.ollama_model``.
            json_mode: When True, request ``format="json"`` from Ollama and
                post-process the reply down to a bare JSON object.
            temperature: Sampling temperature passed via ``options``.
            allow_fallback: When False, never substitute
                ``settings.ollama_fallback_model`` — a missing or failing
                requested model raises ``OllamaError`` instead.

        Returns:
            The cleaned response text. ``self.last_model_used`` records the
            model that actually produced it.

        Raises:
            OllamaError: When every planned model (requested and, if
                allowed, the fallback) fails or is not installed.
        """
        primary = model or settings.ollama_model
        fallback = settings.ollama_fallback_model if allow_fallback else ""

        installed = self.list_models()
        resolved = self._resolve_installed(primary, installed)
        if resolved is not None and resolved != primary:
            logger.info(
                "Resolved requested model '%s' to installed tag '%s'",
                primary, resolved,
            )

        # (model, total attempts) in the order they should be tried.
        plan: list[tuple[str, int]] = []
        if resolved is None:
            logger.warning(
                "Model '%s' is not installed (installed: %s)%s",
                primary, installed,
                f" — going straight to fallback '{fallback}'" if fallback else "",
            )
        else:
            plan.append((resolved, 1 + max(0, self.retries)))
        if fallback and fallback != primary and fallback != resolved:
            plan.append((fallback, 1))

        if not plan:
            raise OllamaError(
                f"Model '{primary}' is not installed on the Ollama server at {self.base_url} "
                + (
                    "and no distinct fallback model is configured."
                    if allow_fallback
                    else "and fallback substitution is disabled for this call."
                )
            )

        last_error: Exception | None = None
        for model_name, attempts in plan:
            for attempt in range(attempts):
                if attempt > 0:
                    delay = float(2 ** (attempt - 1))  # 1s, 2s, 4s, ...
                    logger.info(
                        "Retrying Ollama chat with '%s' in %.0fs (attempt %d/%d)",
                        model_name, delay, attempt + 1, attempts,
                    )
                    time.sleep(delay)
                try:
                    reply = self._chat_once(model_name, prompt, system, json_mode, temperature)
                except OllamaError as exc:
                    last_error = exc
                    logger.warning(
                        "Ollama chat attempt %d/%d with '%s' failed: %s",
                        attempt + 1, attempts, model_name, exc,
                    )
                else:
                    self.last_model_used = model_name
                    return reply
            if len(plan) > 1 and model_name == plan[0][0]:
                logger.warning(
                    "Model '%s' exhausted all attempts — falling back to '%s'",
                    model_name, plan[-1][0],
                )

        tried = ", ".join(f"'{name}'" for name, _ in plan)
        raise OllamaError(
            f"All Ollama chat attempts failed (tried {tried} for requested "
            f"model '{primary}'): {last_error}"
        ) from last_error

    def _chat_once(
        self,
        model: str,
        prompt: str,
        system: str | None,
        json_mode: bool,
        temperature: float,
    ) -> str:
        """Perform one ``POST /api/chat`` call and clean the response.

        Args:
            model: Model name to query.
            prompt: User message content.
            system: Optional system prompt.
            json_mode: Whether to request/extract JSON output.
            temperature: Sampling temperature.

        Returns:
            Cleaned, non-empty response text.

        Raises:
            OllamaError: On any transport, HTTP, payload or empty-response error.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, object] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": settings.ollama_num_ctx},
        }
        if json_mode and not _NO_JSON_FORMAT_RE.search(model):
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed for model '{model}': {exc}") from exc
        except ValueError as exc:  # non-JSON response body
            raise OllamaError(f"Ollama returned a non-JSON body for model '{model}': {exc}") from exc

        if not isinstance(data, dict):
            raise OllamaError(f"Unexpected Ollama response type for model '{model}': {type(data).__name__}")
        if data.get("error"):
            raise OllamaError(f"Ollama server error for model '{model}': {data['error']}")

        message = data.get("message") or {}
        content = str(message.get("content") or "") if isinstance(message, dict) else ""

        text = _strip_think(content)
        if json_mode:
            text = _extract_json_object(text)
        if not text:
            raise OllamaError(f"Empty response from model '{model}' after cleanup")
        return text
