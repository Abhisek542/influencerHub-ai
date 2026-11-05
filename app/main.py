"""HTTP layer for the agents. Each response streams NDJSON events so the UI can show the agent working.

Local demo only: run with --host 127.0.0.1. Paused runs are held in memory and end if the service restarts.
"""
import json
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langgraph.types import Command
from pydantic import BaseModel, Field, ValidationError

from .agents import copilot, scout, shortlist
from .auth import AuthError, User, verify_token
from .backend import BackendClient, BackendError
from .config import settings
from .context import AgentContext
from .llm import LLMError, OllamaLLM

app = FastAPI(title="InfluencerHub AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)

GRAPHS = {
    "shortlist": shortlist.build_graph(),
    "copilot": copilot.build_graph(),
    "scout": scout.build_graph(),
}
ROLES = {"shortlist": "BRAND", "copilot": "BRAND", "scout": "INFLUENCER"}

# Each pause has a schema, so a bad reply is refused before it can reach the graph.
RESUME_SCHEMAS: dict[str, type[BaseModel]] = {
    "approve_shortlist": shortlist.ShortlistResume,
    "question": copilot.CopilotAnswer,
    "approve_campaign": copilot.CopilotApproval,
    "choose_campaign": scout.ScoutChoice,
    "approve_application": scout.ScoutApproval,
}

ERROR_MESSAGES = {
    "session_expired": "Your session has expired. Please log in again.",
    "forbidden": "You don't have permission to do that.",
    "not_found": "That item wasn't found, or it isn't yours.",
    "backend_unavailable": "The InfluencerHub backend isn't reachable.",
    "backend_error": "The InfluencerHub backend reported an error.",
    "model_unavailable": "The local AI model isn't running. The rest of InfluencerHub still works.",
    "model_timeout": "The local AI model took too long. Please try again.",
    "invalid_model_output": "The local AI model returned an unusable answer. Please try again.",
}


@dataclass
class Run:
    agent: str
    owner: str


RUNS: dict[str, Run] = {}


class ShortlistStart(BaseModel):
    campaign_id: int


class CopilotStart(BaseModel):
    brief: str = Field(min_length=1, max_length=4000)


class ResumeBody(BaseModel):
    value: dict[str, Any]


def current_user(authorization: str | None) -> tuple[str, User]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing token")
    token = authorization[7:]
    try:
        return token, verify_token(token, settings.jwt_secret)
    except AuthError as error:
        raise HTTPException(401, "Invalid token") from error


def require_role(user: User, agent: str) -> None:
    if user.role != ROLES[agent]:
        raise HTTPException(403, "This agent isn't available for your role")


def _line(event: dict) -> bytes:
    return (json.dumps(event, default=str) + "\n").encode()


async def _stream(run_id: str, agent: str, token: str, graph_input: Any) -> AsyncIterator[bytes]:
    graph = GRAPHS[agent]
    config = {"configurable": {"thread_id": run_id}}
    backend = BackendClient(token)
    context = AgentContext(backend=backend, llm=OllamaLLM())
    yield _line({"type": "run", "run_id": run_id, "agent": agent})
    try:
        paused = False
        async for mode, chunk in graph.astream(graph_input, config, context=context, stream_mode=["custom", "updates"]):
            if mode == "custom":
                yield _line(chunk)
            elif "__interrupt__" in chunk:
                paused = True
                yield _line({"type": "interrupt", "run_id": run_id, "payload": chunk["__interrupt__"][0].value})
        if not paused:
            state = await graph.aget_state(config)
            yield _line({"type": "done", "run_id": run_id, "result": state.values.get("result")})
            RUNS.pop(run_id, None)
    except (BackendError, LLMError) as error:
        yield _line({"type": "error", "code": error.code, "message": ERROR_MESSAGES.get(error.code, "Something went wrong.")})
    except Exception:
        yield _line({"type": "error", "code": "internal", "message": "The agent hit an unexpected error."})
    finally:
        await backend.aclose()


def _start(agent: str, user: User, token: str, graph_input: dict) -> StreamingResponse:
    run_id = str(uuid.uuid4())
    RUNS[run_id] = Run(agent=agent, owner=user.email)
    return StreamingResponse(_stream(run_id, agent, token, graph_input), media_type="application/x-ndjson")


@app.post("/agents/shortlist/runs")
async def start_shortlist(body: ShortlistStart, authorization: str | None = Header(default=None)):
    token, user = current_user(authorization)
    require_role(user, "shortlist")
    return _start("shortlist", user, token, {"campaign_id": body.campaign_id})


@app.post("/agents/copilot/runs")
async def start_copilot(body: CopilotStart, authorization: str | None = Header(default=None)):
    token, user = current_user(authorization)
    require_role(user, "copilot")
    return _start("copilot", user, token, {"brief": [body.brief], "rounds": 0})


@app.post("/agents/scout/runs")
async def start_scout(authorization: str | None = Header(default=None)):
    token, user = current_user(authorization)
    require_role(user, "scout")
    display_name = user.email.split("@")[0].replace(".", " ").title()
    return _start("scout", user, token, {"display_name": display_name})


@app.post("/agents/runs/{run_id}/resume")
async def resume(run_id: str, body: ResumeBody, authorization: str | None = Header(default=None)):
    token, user = current_user(authorization)
    run = RUNS.get(run_id)
    # The same answer for a missing run and someone else's, so run ids reveal nothing.
    if not run or run.owner != user.email:
        raise HTTPException(404, "Run not found")

    state = await GRAPHS[run.agent].aget_state({"configurable": {"thread_id": run_id}})
    pending = [i for task in state.tasks for i in task.interrupts]
    if not pending:
        raise HTTPException(409, "This run isn't waiting for an answer")

    kind = pending[0].value["kind"]
    try:
        value = RESUME_SCHEMAS[kind].model_validate(body.value).model_dump()
    except ValidationError as error:
        problems = [{"field": ".".join(map(str, e["loc"])), "message": e["msg"]} for e in error.errors()]
        raise HTTPException(422, problems) from error
    if kind in ("approve_campaign", "approve_application") and value["approved"]:
        field = "campaign" if kind == "approve_campaign" else "application"
        if not value[field]:
            raise HTTPException(422, [{"field": field, "message": "Approval needs the details to submit"}])

    return StreamingResponse(_stream(run_id, run.agent, token, Command(resume=value)), media_type="application/x-ndjson")


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3, trust_env=False) as client:
            tags = (await client.get(f"{settings.ollama_url}/api/tags")).json()
        names = [m["name"] for m in tags.get("models", [])]
        return {"status": "ok", "model": settings.ollama_model, "model_ready": settings.ollama_model in names}
    except (httpx.HTTPError, ValueError):
        return {"status": "degraded", "model": settings.ollama_model, "model_ready": False}
