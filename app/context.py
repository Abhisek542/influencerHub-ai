"""Runtime dependencies for a graph run. LangGraph does not checkpoint this, which keeps the token out of saved state."""
from dataclasses import dataclass

from .backend import BackendClient
from .llm import LLM


@dataclass
class AgentContext:
    backend: BackendClient
    llm: LLM


def step(runtime, name: str, message: str) -> None:
    runtime.stream_writer({"type": "step", "step": name, "message": message})
