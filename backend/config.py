"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Council members — using free providers (no OpenRouter needed)
COUNCIL_MODELS = [
    "ollama/qwen2.5:0.5b",                   # Local (Ollama, CPU-only)
    "google/gemini-3.1-flash-lite",           # Google Flash Lite (free, fast)
    "groq/llama-3.3-70b-versatile",          # Groq Llama 3.3 (free, fast)
    "huggingface/llama-3.1-8b",              # HuggingFace Inference (free tier)
    "groq/kimi-k2-instruct",                 # Groq KiMi-K2 1T param (free!)
    "groq/llama-4-scout",                    # Groq Llama 4 Scout (free!)
    "groq/gpt-oss-120b",                     # OpenAI open-weight MoE 120B
    "groq/qwen3-32b",                        # Qwen3 32B reasoning model
]

# Chairman model — synthesizes final response (upgraded to 3.1 Pro)
CHAIRMAN_MODEL = "google/gemini-3.1-pro"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
