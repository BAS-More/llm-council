"""Multi-provider LLM client — calls different APIs based on provider."""

import os
import asyncio
import httpx
from typing import List, Dict, Any, Optional

PROVIDERS = {
    "ollama/llama3.1:8b": {
        "name": "Ollama Llama 3.1",
        "base_url": "http://localhost:11434/v1/chat/completions",
        "api_key": "ollama",  # Ollama doesn't need a real key but accepts any
        "model": "llama3.1:8b",
    },
    "google/gemini-2.5-pro": {
        "name": "Google Gemini 2.5 Pro",
        "type": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "model": "gemini-2.5-pro",
    },
    "google/gemini-2.5-flash": {
        "name": "Google Gemini 2.5 Flash",
        "type": "google",
        "api_key_env": "GOOGLE_API_KEY",
        "model": "gemini-2.5-flash",
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
    "groq/deepseek-r1-distill-llama-70b": {
        "name": "Groq DeepSeek R1 70B",
        "base_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key_env": "GROQ_API_KEY",
        "model": "deepseek-r1-distill-llama-70b",
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
        if provider.get("type") == "google":
            text = await _query_google(provider, messages, timeout)
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

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            provider["base_url"],
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
