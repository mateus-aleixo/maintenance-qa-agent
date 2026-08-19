"""Provider-agnostic LLM access. Three clients, one protocol, no framework.

- StubLLM: deterministic, scriptable. Tests and CI run the full pipeline with it.
- OllamaClient: local models, the development default. Zero marginal cost.
- OpenAICompatClient: any OpenAI-compatible endpoint (OpenAI, Groq, vLLM, ...).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .config import Config


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class TokenDist:
    """Probabilities over the candidate first tokens of a reply.

    The reason this exists: a score the model *writes* is quantised by the model's
    own habits — asking for 0-100 produced three values in practice. The
    distribution over a single next token is continuous by construction and the
    model cannot round it off, which is exactly what conformal calibration needs.
    """

    probs: dict[str, float]

    def mass(self, *variants: str) -> float:
        """Total probability on any spelling of a token (' YES', 'Yes', 'yes')."""
        want = {v.strip().lower() for v in variants}
        return sum(p for t, p in self.probs.items() if t.strip().lower() in want)


class LLMClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult: ...


@dataclass
class StubLLM:
    """Returns scripted responses in order; repeats the last one when exhausted.

    Default script echoes a grounded refusal-safe answer so pipeline tests have a
    predictable shape without any model.
    """

    script: list[str] = field(default_factory=lambda: ["STUB ANSWER [1]"])
    calls: list[tuple[str, str]] = field(default_factory=list)

    dists: list[dict[str, float]] = field(default_factory=list)

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        self.calls.append((system, user))
        idx = min(len(self.calls) - 1, len(self.script) - 1)
        return LLMResult(text=self.script[idx], model="stub")

    def top_logprobs(self, system: str, user: str, top_k: int = 5) -> TokenDist:
        self.calls.append((system, user))
        idx = min(len(self.calls) - 1, len(self.dists) - 1) if self.dists else 0
        return TokenDist(dict(self.dists[idx]) if self.dists else {})


class OllamaClient:
    def __init__(self, cfg: Config):
        self.url = cfg.ollama_url.rstrip("/")
        self.model = cfg.ollama_model

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = httpx.post(
            f"{self.url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
                "options": {"temperature": temperature},
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        return LLMResult(
            text=data["message"]["content"],
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )


class OpenAICompatClient:
    def __init__(self, cfg: Config):
        self.base = cfg.openai_base.rstrip("/")
        self.model = cfg.openai_model
        self.key = cfg.openai_key

    def complete(self, system: str, user: str, temperature: float = 0.0) -> LLMResult:
        resp = httpx.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key}"},
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return LLMResult(
            text=data["choices"][0]["message"]["content"],
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def top_logprobs(self, system: str, user: str, top_k: int = 5) -> TokenDist:
        """Distribution over the first reply token.

        Ollama serves this on its OpenAI-compatible route (`/v1`), not on its
        native `/api/chat` — so point `CRAG_OPENAI_BASE` at
        `http://localhost:11434/v1` to use a local model here.
        """
        resp = httpx.post(
            f"{self.base}/chat/completions",
            headers={"Authorization": f"Bearer {self.key or 'ollama'}"},
            json={
                "model": self.model,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": top_k,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0].get("logprobs", {}).get("content") or []
        if not content:
            return TokenDist({})
        return TokenDist({
            t["token"]: math.exp(t["logprob"]) for t in content[0].get("top_logprobs", [])
        })


def get_llm(cfg: Config) -> LLMClient:
    if cfg.llm_provider == "stub":
        return StubLLM()
    if cfg.llm_provider == "ollama":
        return OllamaClient(cfg)
    if cfg.llm_provider == "openai":
        return OpenAICompatClient(cfg)
    raise ValueError(f"unknown LLM provider: {cfg.llm_provider!r}")
