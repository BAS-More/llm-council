"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# Council members — using free providers (no OpenRouter needed)
COUNCIL_MODELS = [
    "ollama/llama3.1:8b",                    # Local (Ollama, free)
    "google/gemini-2.5-pro",                  # Google AI Studio (free tier)
    "groq/llama-3.3-70b-versatile",          # Groq Llama 3.3 (free, fast)
    "huggingface/llama-3.1-8b",              # HuggingFace Inference (free tier)
    "groq/kimi-k2-instruct",                 # Groq KiMi-K2 1T param (free!)
    "groq/llama-4-scout",                    # Groq Llama 4 Scout (free!)
]

# Chairman model — synthesizes final response
CHAIRMAN_MODEL = "google/gemini-2.5-pro"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
