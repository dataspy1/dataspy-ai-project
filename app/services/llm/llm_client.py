from __future__ import annotations

import json
import time
from typing import Optional, Dict, Any

from app.core.config import get_settings


class LLMClient:
    def __init__(self):
        self.settings = get_settings()

        self.provider = self.settings.validated_llm_provider
        self.allow_fallback = bool(self.settings.llm_allow_fallback)
        self.retry_attempts = max(0, int(self.settings.llm_retry_attempts))
        self.retry_delay_seconds = max(0.0, float(self.settings.llm_retry_delay_seconds))

        self.groq_api_key = (self.settings.groq_api_key or "").strip()
        self.groq_model = (self.settings.groq_model or "").strip() or "llama-3.1-8b-instant"

        self.openai_api_key = (self.settings.openai_api_key or "").strip()
        self.openai_model = (self.settings.openai_model or "").strip() or "gpt-4.1-mini"

        self.client = None
        self.model = None

        if self.provider == "groq":
            self._init_groq_client()
        elif self.provider == "openai":
            self._init_openai_client()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}. Expected one of: groq, openai.")

    def _init_groq_client(self):
        from groq import Groq

        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY is missing in .env")

        self.client = Groq(api_key=self.groq_api_key)
        self.model = self.groq_model

    def _init_openai_client(self):
        from openai import OpenAI

        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is missing in .env")

        self.client = OpenAI(api_key=self.openai_api_key)
        self.model = self.openai_model

    def _validate_prompt_inputs(self, system_prompt: str, user_prompt: str):
        if not system_prompt or not str(system_prompt).strip():
            raise ValueError("system_prompt is required")
        if not user_prompt or not str(user_prompt).strip():
            raise ValueError("user_prompt is required")

    def _extract_groq_text(self, response) -> str:
        try:
            if not response or not getattr(response, "choices", None):
                return ""
            choice = response.choices[0]
            if not choice or not getattr(choice, "message", None):
                return ""
            content = choice.message.content
            return str(content).strip() if content and str(content).strip() else ""
        except Exception:
            return ""

    def _extract_openai_text(self, response) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return str(response.output_text).strip()

        try:
            texts = []
            for item in getattr(response, "output", []) or []:
                for content in getattr(item, "content", []) or []:
                    text_value = getattr(content, "text", None)
                    if text_value:
                        texts.append(str(text_value))
            return "\n".join(texts).strip()
        except Exception:
            return ""

    def _generate_with_groq(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )

        content = self._extract_groq_text(response)
        if not content:
            raise ValueError("Empty response received from Groq")
        return content

    def _generate_with_openai(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            temperature=temperature,
        )

        content = self._extract_openai_text(response)
        if not content:
            raise ValueError("Empty response received from OpenAI")
        return content

    def _generate_with_retry(self, fn, system_prompt: str, user_prompt: str, provider_name: str, temperature: float = 0.2) -> str:
        last_error: Optional[Exception] = None
        total_attempts = self.retry_attempts + 1

        for attempt in range(total_attempts):
            try:
                return fn(system_prompt, user_prompt, temperature)
            except Exception as e:
                last_error = e
                if attempt < total_attempts - 1:
                    time.sleep(self.retry_delay_seconds)

        raise RuntimeError(f"{provider_name} generation failed after {total_attempts} attempt(s): {str(last_error)}") from last_error

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        allow_fallback: Optional[bool] = None,
        temperature: float = 0.2,
    ) -> str:
        self._validate_prompt_inputs(system_prompt, user_prompt)

        use_fallback = self.allow_fallback if allow_fallback is None else bool(allow_fallback)
        primary_provider = self.provider
        primary_model = self.model

        try:
            if primary_provider == "groq":
                return self._generate_with_retry(
                    fn=self._generate_with_groq,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    provider_name=f"Groq | model={primary_model}",
                    temperature=temperature,
                )

            if primary_provider == "openai":
                return self._generate_with_retry(
                    fn=self._generate_with_openai,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    provider_name=f"OpenAI | model={primary_model}",
                    temperature=temperature,
                )

            raise ValueError(f"Unsupported LLM provider: {primary_provider}")

        except Exception as primary_error:
            if use_fallback and primary_provider == "groq" and self.openai_api_key:
                original_client = self.client
                original_model = self.model
                original_provider = self.provider

                try:
                    self.provider = "openai"
                    self._init_openai_client()

                    return self._generate_with_retry(
                        fn=self._generate_with_openai,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        provider_name=f"OpenAI fallback | model={self.model}",
                        temperature=temperature,
                    )

                except Exception as fallback_error:
                    raise RuntimeError(
                        "LLM generation failed on primary provider Groq and fallback OpenAI. "
                        f"Primary error: {str(primary_error)} | Fallback error: {str(fallback_error)}"
                    ) from fallback_error

                finally:
                    self.provider = original_provider
                    self.client = original_client
                    self.model = original_model

            raise RuntimeError(
                f"LLM generation failed | provider={primary_provider} | model={primary_model} | error={str(primary_error)}"
            ) from primary_error

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: Optional[Dict[str, Any]] = None,
        allow_fallback: Optional[bool] = None,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        json_instruction = """
Return valid JSON only.
Do not wrap in markdown.
Do not add explanatory text.
If information is missing, use null, empty list, or an explicit safe note in a field.
"""
        if schema_hint:
            json_instruction += f"\nTarget JSON shape hint:\n{json.dumps(schema_hint, ensure_ascii=False, indent=2)}\n"

        content = self.generate_text(
            system_prompt=system_prompt + "\n" + json_instruction,
            user_prompt=user_prompt,
            allow_fallback=allow_fallback,
            temperature=temperature,
        )

        try:
            return json.loads(content)
        except Exception as exc:
            raise RuntimeError(f"LLM returned non-JSON output: {content}") from exc
