import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from datetime import date, timedelta

import httpx
import pytest
from fastapi.testclient import TestClient
from langgraph.types import Command
from pydantic import ValidationError

from app import main
from app.agents import copilot, scout
from app.auth import AuthError, verify_token
from app.backend import BackendClient
from app.context import AgentContext

SECRET = "test-secret-" + "x" * 64
FUTURE = (date.today() + timedelta(days=30)).isoformat()


def token(email="brand@demo.test", role="BRAND", exp=None, secret=SECRET):
    def b64(data):
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    header = b64(json.dumps({"alg": "HS512"}).encode())
    payload = b64(json.dumps({"sub": email, "role": role, "exp": exp or time.time() + 3600}).encode())
    signature = b64(hmac.new(secret.encode(), f"{header}.{payload}".encode(), hashlib.sha512).digest())
    return f"{header}.{payload}.{signature}"


class ScriptedLLM:
    def __init__(self, *replies):
        self.replies = list(replies)

    async def structured(self, schema, system, user):
        return self.replies.pop(0)


# --- auth


def test_valid_token_is_accepted():
    user = verify_token(token(), SECRET)
    assert (user.email, user.role) == ("brand@demo.test", "BRAND")


@pytest.mark.parametrize(
    "bad", [token(secret="other-secret" + "y" * 64), token(exp=time.time() - 10), "not-a-token", "a.b.c"]
)
def test_bad_tokens_are_refused(bad):
    with pytest.raises(AuthError):
        verify_token(bad, SECRET)


# --- copilot


@pytest.mark.parametrize(
    "text,expected",
    [("50k", "50000"), ("Rs 50,000", "50000"), ("1.5 lakh", "150000"), ("2 crore", "20000000"), ("750", "750"), ("lots", None)],
)
def test_budget_parsing(text, expected):
    assert copilot.parse_budget(text) == expected


def test_invented_brand_budget_and_deadline_are_treated_as_missing():
    extraction = copilot.CampaignExtraction(title="T", description="D", brand_name="MadeUpCo", niche="Fitness", budget="50000")
    fields, missing = copilot.review_extraction(extraction, "Need fitness creators soon, 3 videos")
    assert set(missing) == {"brandName", "budget", "deadline"}
    assert fields["brandName"] == "" and fields["budget"] == "" and fields["deadline"] == ""


def test_the_text_null_is_not_a_brand():
    assert copilot.find_brand("null", "budget is null for now") == ""


def test_brand_is_found_when_the_model_misses_it():
    assert copilot.find_brand(None, "GlowFuel is launching a bar") == "GlowFuel"
    assert copilot.find_brand(None, "Need creators for DewLeaf's new sunscreen") == "DewLeaf"
    assert copilot.find_brand(None, "Budget is 500 and we need creators") == ""


TODAY = date(2026, 11, 5)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("closes on 15 December", "2026-12-15"),
        ("deadline December 15th, 2027", "2027-12-15"),
        ("Dec 1", "2026-12-01"),
        ("15 March", "2027-03-15"),
        ("by 2026-12-20", "2026-12-20"),
        ("in 3 weeks", "2026-11-26"),
        ("in 2 months", "2027-01-05"),
        ("next month", "2026-12-05"),
        ("closes 15 December. Actually make it 20 December", "2026-12-20"),
        ("2020-01-01", None),
        ("soon", None),
        ("3 videos and 2 stories", None),
        ("31 February", None),
    ],
)
def test_deadline_parsing(text, expected):
    assert copilot.parse_deadline(text, TODAY) == expected


@pytest.mark.parametrize("change", [{"budget": "abc"}, {"budget": "0"}, {"deadline": "soon"}, {"deadline": "2020-01-01"}, {"title": ""}])
def test_campaign_proposal_refuses_values_that_break_the_backend(change):
    valid = {"title": "T", "description": "D", "brandName": "B", "niche": "N", "budget": "100", "deadline": FUTURE}
    copilot.CampaignProposal(**valid)
    with pytest.raises(ValidationError):
        copilot.CampaignProposal(**{**valid, **change})


class CampaignBackend:
    def __init__(self):
        self.created = []

    def __call__(self, request):
        assert request.url.path == "/api/campaign/create"
        self.created.append(json.loads(request.content))
        return httpx.Response(200, text="Campaign created successfully!")


def test_copilot_asks_then_creates_only_after_approval():
    fake = CampaignBackend()
    incomplete = copilot.CampaignExtraction(title="T", description="D", brand_name="Glow", niche="Fitness")
    complete = incomplete.model_copy(update={"budget": "50k"})
    graph = copilot.build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    context = AgentContext(BackendClient("t", transport=httpx.MockTransport(fake)), ScriptedLLM(incomplete, complete))
    campaign = {"title": "T", "description": "D", "brandName": "Glow", "niche": "Fitness", "budget": "50000", "deadline": FUTURE}

    async def run():
        first = await graph.ainvoke({"brief": ["Glow needs fitness creators"], "rounds": 0}, config, context=context)
        assert first["__interrupt__"][0].value["kind"] == "question"
        second = await graph.ainvoke(Command(resume={"answer": f"Budget 50k, deadline {FUTURE}"}), config, context=context)
        assert second["__interrupt__"][0].value["kind"] == "approve_campaign"
        assert second["__interrupt__"][0].value["proposal"]["deadline"] == FUTURE
        assert second["__interrupt__"][0].value["proposal"]["budget"] == "50000"
        assert fake.created == []
        return await graph.ainvoke(Command(resume={"approved": True, "campaign": campaign}), config, context=context)

    assert asyncio.run(run())["result"]["status"] == "DONE"
    assert fake.created == [campaign]


def test_copilot_declined_creates_nothing():
    fake = CampaignBackend()
    extraction = copilot.CampaignExtraction(title="T", description="D", brand_name="Glow", niche="Fitness", budget="500")
    graph = copilot.build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    context = AgentContext(BackendClient("t", transport=httpx.MockTransport(fake)), ScriptedLLM(extraction))

    async def run():
        await graph.ainvoke({"brief": [f"Glow fitness, budget 500, closes {FUTURE}"], "rounds": 0}, config, context=context)
        return await graph.ainvoke(Command(resume={"approved": False, "campaign": None}), config, context=context)

    assert asyncio.run(run())["result"]["status"] == "DECLINED"
    assert fake.created == []


# --- scout


class ScoutBackend:
    def __init__(self, apply_message="Application has been applied", bio="Vegan home-workout creator"):
        self.applied = []
        self.apply_message = apply_message
        self.bio = bio

    def __call__(self, request):
        path = request.url.path
        if path == "/api/influencer/profile/me":
            return httpx.Response(200, json={"bio": self.bio, "socialLinks": "instagram.com/x"})
        if path == "/api/campaign/filtr":
            return httpx.Response(
                200,
                json={
                    "content": [
                        {"id": 1, "title": "Protein", "niche": "Fitness", "description": "D", "brandName": "G", "deadline": FUTURE, "applied": False},
                        {"id": 2, "title": "Done", "niche": "Fitness", "description": "D", "brandName": "G", "deadline": FUTURE, "applied": True},
                        {"id": 3, "title": "Old", "niche": "Fitness", "description": "D", "brandName": "G", "deadline": "2020-01-01", "applied": False},
                    ]
                },
            )
        if path == "/api/influencer/apply/1":
            self.applied.append(json.loads(request.content))
            return httpx.Response(200, json={"message": self.apply_message})
        return httpx.Response(404)


def scout_run(fake, llm):
    graph = scout.build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    return graph, config, AgentContext(BackendClient("t", transport=httpx.MockTransport(fake)), llm)


GOOD_MATCH = scout.CampaignMatch(fit=5, reason="Fits", profile_quote="home-workout creator")
APPLICATION = {"message": "Hi", "influencerName": "Meera", "niche": "Fitness", "platform": "Instagram"}


def run_scout_to_apply(fake):
    graph, config, context = scout_run(fake, ScriptedLLM(GOOD_MATCH, scout.Pitch(message="I would love to", niche="Fitness")))

    async def run():
        first = await graph.ainvoke({"display_name": "Meera"}, config, context=context)
        value = first["__interrupt__"][0].value
        assert value["kind"] == "choose_campaign"
        assert [m["campaign"]["id"] for m in value["matches"]] == [1]  # applied and expired campaigns are excluded
        second = await graph.ainvoke(Command(resume={"campaign_id": 1}), config, context=context)
        draft = second["__interrupt__"][0].value
        assert draft["kind"] == "approve_application" and draft["draft"]["platform"] == "Instagram"
        assert fake.applied == []
        return await graph.ainvoke(Command(resume={"approved": True, "application": APPLICATION}), config, context=context)

    return asyncio.run(run())


def test_scout_applies_only_after_approval():
    fake = ScoutBackend()
    assert run_scout_to_apply(fake)["result"]["status"] == "DONE"
    assert fake.applied == [APPLICATION]


def test_scout_reports_a_duplicate_even_though_the_backend_returns_200():
    fake = ScoutBackend(apply_message="Your campaign has been already applied")
    assert run_scout_to_apply(fake)["result"]["status"] == "ALREADY_APPLIED"


def test_match_without_a_real_quote_is_not_offered():
    fake = ScoutBackend()
    unsupported = scout.CampaignMatch(fit=5, reason="r", profile_quote="award-winning athlete")
    graph, config, context = scout_run(fake, ScriptedLLM(unsupported))
    final = asyncio.run(graph.ainvoke({"display_name": "M"}, config, context=context))
    assert final["result"]["status"] == "NO_MATCHES"


def test_scout_without_a_bio_makes_no_model_call():
    fake = ScoutBackend(bio="")
    graph, config, context = scout_run(fake, ScriptedLLM())
    final = asyncio.run(graph.ainvoke({"display_name": "M"}, config, context=context))
    assert final["result"]["status"] == "NEEDS_PROFILE"


def test_invented_follower_count_in_a_pitch_gets_a_warning():
    assert scout.invented_claim_warnings("I have 50k followers", "I make videos")
    assert not scout.invented_claim_warnings("I make videos", "I make videos")


# --- HTTP layer


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main.settings, "jwt_secret", SECRET)
    main.RUNS.clear()
    return TestClient(main.app)


def bearer(**kwargs):
    return {"Authorization": f"Bearer {token(**kwargs)}"}


def test_missing_or_bad_token_is_refused(client):
    assert client.post("/agents/scout/runs").status_code == 401
    assert client.post("/agents/scout/runs", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_wrong_role_is_refused(client):
    headers = bearer(email="i@demo.test", role="INFLUENCER")
    assert client.post("/agents/shortlist/runs", json={"campaign_id": 1}, headers=headers).status_code == 403
    assert client.post("/agents/copilot/runs", json={"brief": "x"}, headers=headers).status_code == 403
    assert client.post("/agents/scout/runs", headers=bearer()).status_code == 403


def test_another_users_run_cannot_be_resumed(client):
    main.RUNS["run-1"] = main.Run(agent="shortlist", owner="brand@demo.test")
    response = client.post("/agents/runs/run-1/resume", json={"value": {}}, headers=bearer(email="other@demo.test"))
    assert response.status_code == 404


def test_cors_allows_only_the_local_frontend(client):
    def preflight(origin):
        return client.options(
            "/agents/scout/runs",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"},
        )

    assert preflight("http://localhost:4200").headers.get("access-control-allow-origin") == "http://localhost:4200"
    assert "access-control-allow-origin" not in preflight("https://evil.example").headers
