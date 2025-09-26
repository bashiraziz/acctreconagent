from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import google.generativeai as genai


@dataclass
class GeminiConfig:
    """Configuration required to talk to Gemini."""

    api_key: str | None = None
    model: str = "gemini-1.5-flash"
    safety_settings: dict[str, Any] | None = None
    generation_config: dict[str, Any] | None = None

    def resolve_api_key(self) -> str:
        key = self.api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError(
                "Gemini API key not provided. Set GEMINI_API_KEY/GOOGLE_API_KEY or pass api_key explicitly."
            )
        return key


class GeminiLLM:
    """Thin wrapper around Google Gemini for text generation."""

    def __init__(self, config: GeminiConfig | None = None) -> None:
        self.config = config or GeminiConfig()
        api_key = self.config.resolve_api_key()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=self.config.model,
            safety_settings=self.config.safety_settings,
            generation_config=self.config.generation_config,
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = self.model.generate_content(prompt, **kwargs)
        if not response.parts:
            return ""
        return "".join(part.text or "" for part in response.parts)
