"""Structured output from the local Qwen model through Ollama. Every reply is validated with Pydantic."""
import asyncio
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .config import settings

T = TypeVar("T", bound=BaseModel)

# A laptop GPU runs one generation well, so concurrent requests wait their turn.
_model_lock = asyncio.Lock()


class LLMError(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


class LLM(Protocol):
    async def structured(self, schema: type[T], system: str, user: str) -> T: ...


def as_data(label: str, text: str | None) -> str:
    """Wraps untrusted text so the prompt can treat it as data. It cannot close its own delimiter."""
    clean = (text or "").replace("<", "(").replace(">", ")")
    return f"<{label}>\n{clean}\n</{label}>"


class OllamaLLM:
    def __init__(self, transport: httpx.AsyncBaseTransport | None = None, retries: int = 1):
        self._transport = transport
        self._retries = retries

    async def structured(self, schema: type[T], system: str, user: str) -> T:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(self._retries + 1):
            content = await self._chat(messages, schema)
            try:
                return schema.model_validate_json(content)
            except ValidationError as error:
                problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()[:5])
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That JSON was invalid ({problems}). Reply again with valid JSON."},
                ]
        raise LLMError("invalid_model_output")

    async def _chat(self, messages: list[dict], schema: type[BaseModel]) -> str:
        body = {
            "model": settings.ollama_model,
            "messages": messages,
            "stream": False,
            "format": schema.model_json_schema(),
            "keep_alive": "30m",
            # A fixed seed and no temperature, so the same application gets the same assessment on every run.
            "options": {"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 700},
        }
        try:
            async with _model_lock:
                async with httpx.AsyncClient(
                    timeout=settings.model_timeout_seconds, transport=self._transport, trust_env=False
                ) as client:
                    response = await client.post(f"{settings.ollama_url}/api/chat", json=body)
                    response.raise_for_status()
                    return response.json()["message"]["content"]
        except httpx.TimeoutException as error:
            raise LLMError("model_timeout") from error
        except (httpx.HTTPError, KeyError, ValueError) as error:
            raise LLMError("model_unavailable") from error
