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
    "anthropic/claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6",
        "type": "anthropic",
        "api_key_env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-4-6",
    },
    "openai/gpt-5.4": {
        "name": "GPT-5.4",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4",
    },
    "openai/gpt-5.4-mini": {
        "name": "GPT-5.4 Mini",
        "base_url": "https://api.openai.com/v1/chat/completions",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-5.4-mini",
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

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            base_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={
                "model": provider["model"],
                "messages": messages,
                "max_tokens": 4096,
                "temperature": 0.7,
            },
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
        "max_tokens": 4096,
        "messages": api_messages,
    }
    if system_text:
        body["system"] = system_text

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
        # Anthropic returns content as array of blocks
        return "".join(
            block["text"] for block in data["content"] if block["type"] == "text"
        )


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
