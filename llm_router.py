#!/usr/bin/env python3
"""
LLM Router — automatic fallback: Groq -> Gemini -> Cerebras
Simplified version of trading-bot's llm_router.py for Polymarket bot.

Usage:
    from llm_router import call_llm
    text, provider, model = call_llm(prompt, max_tokens=400)
"""

import json
import logging
import os
import urllib.request

log = logging.getLogger(__name__)

PROVIDERS = [
    {
        "name": "gemini",
        "env_key": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}",
        "model": "gemini-2.5-flash",
        "fallback_model": "gemini-2.0-flash",
        "format": "gemini",
    },
    {
        "name": "groq",
        "env_key": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama-3.3-70b-versatile",
        "fallback_model": "llama-3.1-8b-instant",
        "format": "openai",
    },
    {
        "name": "cerebras",
        "env_key": "CEREBRAS_API_KEY",
        "url": "https://api.cerebras.ai/v1/chat/completions",
        "model": "llama3.1-70b",
        "format": "openai",
    },
    {
        "name": "openrouter",
        "env_key": "OPEN_ROUTER",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "meta-llama/llama-4-maverick:free",
        "fallback_model": "google/gemini-2.0-flash-exp:free",
        "format": "openai",
    },
]


def _call_openai_format(url, api_key, model, prompt, max_tokens, temperature):
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "polymarket-bot/1.0",
    }
    if "openrouter.ai" in url:
        headers["HTTP-Referer"] = "https://github.com/polymarket-bot"
    req = urllib.request.Request(url, data=payload, headers=headers)
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def _call_gemini_format(url, api_key, model, prompt, max_tokens, temperature):
    full_url = url.format(model=model, api_key=api_key)
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
        "tools": [{"google_search": {}}],
    }).encode()
    req = urllib.request.Request(full_url, data=payload, headers={
        "Content-Type": "application/json",
        "User-Agent": "polymarket-bot/1.0",
    })
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    parts = data["candidates"][0]["content"]["parts"]
    text_parts = [p["text"] for p in parts if "text" in p]
    if not text_parts:
        raise ValueError("Gemini returned no text parts")
    return "\n".join(text_parts)


def call_llm(prompt, max_tokens=400, temperature=0.3):
    """
    Call LLM with automatic fallback chain.
    Returns: (response_text, provider_name, model_name)
    Raises: RuntimeError if ALL providers fail.
    """
    errors = []

    for provider in PROVIDERS:
        api_key = os.environ.get(provider["env_key"])
        if not api_key:
            continue

        models_to_try = [provider["model"]]
        if "fallback_model" in provider:
            models_to_try.append(provider["fallback_model"])

        for model in models_to_try:
            try:
                if provider["format"] == "openai":
                    content = _call_openai_format(
                        provider["url"], api_key, model,
                        prompt, max_tokens, temperature,
                    )
                elif provider["format"] == "gemini":
                    content = _call_gemini_format(
                        provider["url"], api_key, model,
                        prompt, max_tokens, temperature,
                    )
                else:
                    continue

                log.info(f"LLM call OK: {provider['name']}/{model}")
                return content, provider["name"], model

            except Exception as e:
                error_msg = str(e)[:200]
                errors.append(f"{provider['name']}/{model}: {error_msg}")
                log.warning(f"LLM call failed {provider['name']}/{model}: {error_msg}")

                # Rate limited -> try fallback model, then next provider
                if "429" in error_msg or "rate" in error_msg.lower():
                    continue
                # Other error -> skip to next provider
                else:
                    break

    raise RuntimeError(f"All LLM providers failed:\n" + "\n".join(errors))
