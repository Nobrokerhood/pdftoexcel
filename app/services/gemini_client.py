import logging
import json
import time
from typing import Any

from google import genai
from google.genai import types

from app.core.config import Settings
from app.core.errors import ServiceNotConfiguredError


logger = logging.getLogger(__name__)


class GeminiDocumentClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        if not self.settings.gemini_api_key:
            raise ServiceNotConfiguredError("GEMINI_API_KEY is not configured.")

        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        logger.info("Gemini client configured for model: %s", self.settings.gemini_model)
        return self._client

    def generate_content(self, prompt_parts: list[Any], retries: int = 2):
        client = self._get_client()
        for attempt in range(retries):
            try:
                return client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=prompt_parts,
                )
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2)

    def generate_json(self, prompt_parts: list[Any], retries: int = 2):
        client = self._get_client()
        for attempt in range(retries):
            try:
                response = client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=prompt_parts,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    ),
                )
                text = response.text.strip()
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    cleaned = text.replace("```json", "").replace("```", "").strip()
                    return json.loads(cleaned)
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2)
