"""Thin LLM client. Supports Groq, Gemini and Ollama through plain HTTP requests."""
import time

import requests

from app import config


class LLMError(RuntimeError):
    pass


def _post(url: str, headers: dict, payload: dict, retries: int = 4) -> dict:
    """POST with retry/backoff for rate limits (free tiers hit 429 often)."""
    for attempt in range(retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(f"Could not connect to {url.split('/')[2]}: {exc}") from exc
        if resp.status_code in (429, 500, 502, 503):
            time.sleep(2 ** (attempt + 1))
            continue
        if not resp.ok:
            raise LLMError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()
    raise LLMError("LLM API keeps failing (rate limit?). Wait a minute and retry.")


def _groq(prompt: str, system: str, temperature: float) -> str:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set. Add it to your .env file.")
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    data = _post(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {config.GROQ_API_KEY}"},
        {"model": config.GROQ_MODEL, "messages": messages, "temperature": temperature},
    )
    return data["choices"][0]["message"]["content"]


def _gemini(prompt: str, system: str, temperature: float) -> str:
    if not config.GEMINI_API_KEY:
        raise LLMError("GEMINI_API_KEY is not set. Add it to your .env file.")
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }
    if system:
        payload["system_instruction"] = {"parts": [{"text": system}]}
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{config.GEMINI_MODEL}:generateContent",
        {"x-goog-api-key": config.GEMINI_API_KEY},
        payload,
    )
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise LLMError(f"Gemini returned no text (blocked or empty): {data}") from exc


def _ollama(prompt: str, system: str, temperature: float) -> str:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    data = _post(
        f"{config.OLLAMA_URL}/api/chat",
        {},
        {
            "model": config.OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    return data["message"]["content"]


def generate(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    providers = {"groq": _groq, "gemini": _gemini, "ollama": _ollama}
    try:
        fn = providers[config.LLM_PROVIDER]
    except KeyError:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{config.LLM_PROVIDER}'. Use groq, gemini or ollama."
        ) from None
    return fn(prompt, system, temperature)
