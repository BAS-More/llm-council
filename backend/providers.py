"""Multi-provider LLM client — calls different APIs based on provider."""

import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional

PROVIDERS = {
    "ollama/qwen2.5:0.5b": {
        "name": "Ollama Qwen 2.5 0.5B",
        "base_url_env": "OLLAMA_BASE_URL",
        "base_url_default": "http://localhost:11434",
        "api_key": "ollama",  # Ollama doesn't need a real key but accepts any
        "model": "qwen2.5:0.5b",
    },
    "google/gemini-3.1-pro": {
        "name": "Google Gemini 3.1 Pro",
        "type": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "model": "gemini-3.1-pro-preview",
    },
    "google/gemini-3.1-flash-lite": {
        "name": "Google Gemini 3.1 Flash Lite",
        "type": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "model": "gemini-3.1-flash-lite-preview",
    },
    "google/nano-banana-pro": {
        "name": "Nano Banana Pro (Gemini 3 Pro Image)",
        "type": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "model": "gemini-3-pro-image-preview",
    },
    "google/gemini-3.1-flash-live": {
        "name": "Gemini 3.1 Flash Live (Voice)",
        "type": "google-live",
        "api_key_env": "GOOGLE_API_KEY",
        "model": "gemini-3.1-flash-live-preview",
    },
    "google/veo-3.1": {
        "name": "Veo 3.1 (Video Generation)",
        "type": "google-video",
        "api_key_env": "GOOGLE_API_KEY",
        "model": "veo-3.1-generate-preview",
    },
    "groq/llama-3.3-70b-versatile": {
        "name": "Groq Llama 3.3 70B",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
    },
    "huggingface/llama-3.1-8b": {
        "name": "HuggingFace Llama 3.1",
        "base_url": "https://router.huggingface.co/novita/v3/openai/chat/completions",
        "api_key_env": "HF_API_KEY",
        "model": "meta-llama/llama-3.1-8b-instruct",
    },
    "groq/kimi-k2-instruct": {
        "name": "Groq KiMi-K2 1T",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model": "moonshotai/kimi-k2-instruct",
    },
    "groq/llama-4-scout": {
        "name": "Groq Llama 4 Scout",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
    },
    "groq/gpt-oss-120b": {
        "name": "GPT-OSS 120B (OpenAI Open-Weight)",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model": "openai/gpt-oss-120b",
    },
    "groq/qwen3-32b": {
        "name": "Qwen3 32B",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model": "qwen/qwen3-32b",
    },
    "anthropic/claude-opus-4-6": {
        "name": "Claude Opus 4.6",
        "type": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-opus-4-6",
    },
    "anthropic/claude-opus-4-8": {
        # Council Chairman. IMPORTANT: Opus 4.8 (and 4.7) REJECT manual extended thinking
        # ({"type":"enabled","budget_tokens":...}) with HTTP 400 — they think ADAPTIVELY at
        # effort=high BY DEFAULT on the Messages API. So we send NO thinking block; the
        # high-effort reasoning ("ultrathink") happens automatically. Confirmed against the
        # live platform.claude.com models + extended-thinking docs after a real 400 in prod.
        "name": "Claude Opus 4.8 (adaptive high-effort chairman)",
        "type": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-opus-4-8",
        "max_tokens": 32000,
    },
    "anthropic/claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6",
        "type": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
    "openai/gpt-5.2": {
        "name": "GPT-5.2",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.2-2025-12-11",
    },
    "openai/gpt-5.1-chat": {
        "name": "GPT-5.1 Chat",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.1-chat-latest",
    },
}


async def query_model(
    model_id: str,
    messages: List[Dict[str, str]],
    timeout: float = 120.0,
) -> Optional[Dict[str, Any]]:
    """
    Query a model by its provider/model ID.

    Returns:
        Response dict with 'content' key (matching the old OpenRouter interface),
        or None if failed.
    """
    provider = PROVIDERS.get(model_id)
    if not provider:
        print(f"Unknown model: {model_id}")
        return None

    try:
        ptype = provider.get("type", "")
        if ptype == "google":
            text = await _query_google(provider, messages, timeout)
        elif ptype == "anthropic":
            text = await _query_anthropic(provider, messages, timeout)
        else:
            text = await _query_openai_compatible(provider, messages, timeout)

        if text is None:
            return None

        return {"content": text}

    except Exception as e:
        print(f"Error querying {model_id}: {e}")
        return None


async def _query_openai_compatible(
    provider: dict, messages: list, timeout: float
) -> Optional[str]:
    """Call OpenAI-compatible endpoint (Groq, Ollama, HuggingFace)."""
    api_key = provider.get("api_key") or os.getenv(
        provider.get("api_key_env", ""), ""
    )

    # Resolve base URL: env var override → explicit base_url → default
    if provider.get("base_url_env"):
        base = os.getenv(provider["base_url_env"], provider.get("base_url_default", "http://localhost:11434"))
        base_url = f"{base}/v1/chat/completions"
    else:
        base_url = provider["base_url"]

    # GPT-5.x reasoning models require max_completion_tokens (not max_tokens)
    # and reject custom temperature values — detect by model name
    model_name = provider["model"]
    is_gpt5_family = model_name.startswith("gpt-5") or model_name.startswith("gpt-5.")

    body: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    if is_gpt5_family:
        # GPT-5.x: needs higher budget because reasoning tokens consume the allowance
        # before any visible output is emitted
        body["max_completion_tokens"] = 4096
    else:
        body["max_tokens"] = 4096
        body["temperature"] = 0.7

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json=body,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


async def _query_google(
    provider: dict, messages: list, timeout: float
) -> Optional[str]:
    """Call Google Gemini API (different format from OpenAI)."""
    api_key = os.getenv(provider.get("api_key_env", ""), "")
    model = provider["model"]

    # Convert OpenAI messages format to Google format
    contents = []
    system_text = ""
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    body: Dict[str, Any] = {"contents": contents}
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}

    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=body)
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]


async def _query_anthropic(
    provider: dict, messages: list, timeout: float
) -> Optional[str]:
    """Call Anthropic Messages API (different format from OpenAI)."""
    api_key = os.getenv(provider.get("api_key_env", ""), "")
    model = provider["model"]

    # Separate system message from conversation messages
    system_text = ""
    api_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            api_messages.append({"role": msg["role"], "content": msg["content"]})

    body: Dict[str, Any] = {
        "model": model,
        "max_tokens": provider.get("max_tokens", 4096),
        "messages": api_messages,
    }
    if system_text:
        body["system"] = system_text

    # Manual extended thinking for models that SUPPORT it (Sonnet 4.5 / Opus 4.6 & earlier).
    # (Opus 4.7/4.8 reject manual thinking with 400 — they think adaptively at high effort
    #  by default, so their PROVIDERS entry deliberately has no "thinking" key.)
    # The API requires max_tokens > budget_tokens and the default temperature (we set
    # neither temperature nor a beta header — 16k budget needs no beta). Thinking blocks
    # come back as separate content blocks; the text-only parser below skips them.
    if provider.get("thinking"):
        budget = int(provider.get("thinking_budget", 16000))
        if body["max_tokens"] <= budget:
            body["max_tokens"] = budget + 4096
        body["thinking"] = {"type": "enabled", "budget_tokens": budget}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=body,
        )
        response.raise_for_status()
        data = response.json()
        # Anthropic returns content as an array of blocks; extended-thinking blocks are skipped.
        text = "".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        )
        # Defensive: if the model emitted only thinking blocks and no visible text (whole token
        # budget spent thinking, or truncated mid-think), treat it as a failure so the caller's
        # fallback engages (e.g. the Opus->Sonnet chairman fallback) instead of a silently-empty
        # synthesis. An empty council answer is useless, so None is the right signal here.
        return text or None


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, str]],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Returns:
        Dict mapping model identifier to response dict (or None if failed).
        Matches the old OpenRouter interface exactly.
    """
    tasks = [query_model(mid, messages) for mid in models]
    responses = await asyncio.gather(*tasks)
    return {model: response for model, response in zip(models, responses)}
