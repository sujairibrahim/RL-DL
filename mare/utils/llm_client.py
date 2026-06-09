"""
remarl/mare/utils/llm_client.py
--------------------------------
Thin LLM client that wraps NVIDIA Build API (cloud), Ollama (local),
OpenAI, and Anthropic. MARE agents call this — it is not RL-specific.

Default provider is "nvidia" which requires NVIDIA_API_KEY in the environment.
Ollama provider requires Ollama running locally at http://localhost:11434.
"""

import os
import logging
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Simple LLM wrapper used by MARE agents.

    Args:
        provider:    "nvidia" | "ollama" | "openai" | "anthropic"
        model:       model string e.g. "meta/llama-3.1-8b-instruct", "llama3.1:8b"
        temperature: generation temperature
        max_tokens:  max output tokens
        base_url:    API base URL (NVIDIA or Ollama endpoint)
    """

    def __init__(
        self,
        provider: str = "nvidia",
        model: str = "meta/llama-3.1-8b-instruct",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ):
        self.provider    = provider
        self.model       = model
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.base_url    = base_url.rstrip("/")
        self._client     = None

    # ── Public API ────────────────────────────────────────────────────────

    def call(self, prompt: str, system_prompt: str = "") -> str:
        """Send a prompt and return the response text."""
        if self.provider == "nvidia":
            return self._call_nvidia(prompt, system_prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt, system_prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt, system_prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system_prompt)
        else:
            raise ValueError(
                f"Unknown provider: '{self.provider}'. "
                "Use 'nvidia', 'ollama', 'openai', or 'anthropic'."
            )

    # ── NVIDIA Build API (OpenAI-compatible) ──────────────────────────────

    def _call_nvidia(self, prompt: str, system_prompt: str = "") -> str:
        """Call NVIDIA Build API via the OpenAI-compatible /v1/chat/completions endpoint."""
        import time
        import openai
        from openai import APIConnectionError, APITimeoutError, RateLimitError

        if self._client is None:
            api_key = os.environ.get("NVIDIA_API_KEY")
            if not api_key:
                raise RuntimeError(
                    "NVIDIA_API_KEY is not set.\n"
                    "  → Add it to your .env file:  NVIDIA_API_KEY=nvapi-...\n"
                    "  → Get a key at: https://build.nvidia.com"
                )
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=120.0,
                max_retries=4,
            )

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_attempts = 3
        backoff = 5
        last_exc = None
        for attempt in range(max_attempts):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    timeout=120.0,
                )
                return response.choices[0].message.content.strip()
            except (APIConnectionError, APITimeoutError, RateLimitError) as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"NVIDIA call failed (attempt {attempt+1}/{max_attempts}): {e}. "
                        f"Retrying in {backoff}s..."
                    )
                    time.sleep(backoff)
                    backoff *= 2
                else:
                    logger.error(
                        f"NVIDIA call failed after {max_attempts} attempts: {e}"
                    )
                    raise

        raise RuntimeError(f"unreachable; last_exc={last_exc}")

    # ── Ollama ────────────────────────────────────────────────────────────

    def _call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """Call a local Ollama model via the /api/chat endpoint."""
        url = f"{self.base_url}/api/chat"
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama at "
                f"{self.base_url}.\n"
                "  → Start Ollama with:  ollama serve\n"
                f"  → Pull the model with: ollama pull {self.model}"
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}\nResponse: {response.text}")

    # ── OpenAI (fallback) ─────────────────────────────────────────────────

    def _call_openai(self, prompt: str, system_prompt: str = "") -> str:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY")
            )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    # ── Anthropic (fallback) ──────────────────────────────────────────────

    def _call_anthropic(self, prompt: str, system_prompt: str = "") -> str:
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY")
            )
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()

    # ── Health checks ─────────────────────────────────────────────────────

    @staticmethod
    def check_ollama_health(base_url: str = "http://localhost:11434") -> bool:
        """Return True if Ollama is reachable. Kept for backward compatibility."""
        try:
            r = requests.get(f"{base_url}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.exceptions.ConnectionError:
            logger.error("Ollama is not running. Start it with:  ollama serve")
            return False

    @staticmethod
    def check_nvidia_health() -> bool:
        """Return True if NVIDIA_API_KEY is present and looks valid."""
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key or not api_key.startswith("nvapi-"):
            logger.error(
                "NVIDIA_API_KEY is not set or invalid.\n"
                "  → Add NVIDIA_API_KEY=nvapi-... to your .env file\n"
                "  → Get a key at: https://build.nvidia.com"
            )
            return False
        return True

    @staticmethod
    def check_provider_health(provider: str, base_url: str = "http://localhost:11434") -> bool:
        """Unified health check — dispatches by provider name."""
        if provider == "nvidia":
            return LLMClient.check_nvidia_health()
        elif provider == "ollama":
            return LLMClient.check_ollama_health(base_url)
        return True  # openai/anthropic: key presence checked at call time

    # ── Factory ───────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, config: dict) -> "LLMClient":
        llm_cfg = config.get("llm", {})
        return cls(
            provider=llm_cfg.get("provider", "nvidia"),
            model=llm_cfg.get("model", "meta/llama-3.1-8b-instruct"),
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 2048),
            base_url=llm_cfg.get("base_url", "https://integrate.api.nvidia.com/v1"),
        )

    @classmethod
    def for_agent(cls, agent_role: str, config: dict) -> "LLMClient":
        """Create an LLMClient using the per-agent model from config."""
        llm_cfg = config.get("llm", {})
        agent_models = llm_cfg.get("agent_models", {})
        model = agent_models.get(agent_role, llm_cfg.get("model", "meta/llama-3.1-8b-instruct"))
        return cls(
            provider=llm_cfg.get("provider", "nvidia"),
            model=model,
            temperature=llm_cfg.get("temperature", 0.2),
            max_tokens=llm_cfg.get("max_tokens", 2048),
            base_url=llm_cfg.get("base_url", "https://integrate.api.nvidia.com/v1"),
        )
