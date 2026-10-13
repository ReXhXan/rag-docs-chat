"""Pluggable LLM clients. OpenAI and Groq share the OpenAI-compatible API; Ollama runs locally."""
import httpx


class LLM:
    def generate(self, system: str, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleLLM(LLM):
    def __init__(self, model: str, api_key: str, base_url: str | None = None):
        from openai import OpenAI
        self.client, self.model = OpenAI(api_key=api_key, base_url=base_url), model

    def generate(self, system, prompt):
        r = self.client.chat.completions.create(
            model=self.model, temperature=0.1,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""


class OllamaLLM(LLM):
    def __init__(self, model: str, base_url: str):
        self.model, self.url = model, base_url.rstrip("/") + "/api/chat"

    def generate(self, system, prompt):
        r = httpx.post(self.url, timeout=120, json={
            "model": self.model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        })
        r.raise_for_status()
        return r.json()["message"]["content"]


class MockLLM(LLM):
    """Deterministic stand-in for tests / offline demos: echoes the top source."""

    def generate(self, system, prompt):
        first = prompt.split("[1]", 1)[-1].split("\n[2]", 1)[0].strip()
        return f"Based on the documents: {first[:300]} [1]"


def get_llm(settings) -> LLM:
    p = settings.llm_provider.lower()
    if p == "openai":
        return OpenAICompatibleLLM(settings.llm_model, settings.openai_api_key)
    if p == "groq":
        return OpenAICompatibleLLM(settings.llm_model, settings.groq_api_key,
                                   base_url="https://api.groq.com/openai/v1")
    if p == "ollama":
        return OllamaLLM(settings.llm_model, settings.ollama_base_url)
    return MockLLM()
