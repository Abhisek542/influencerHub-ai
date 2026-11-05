"""Runs the whole demo story against the live local services, then resets the demo data.

Copilot creates a campaign -> Scout finds it and applies -> Shortlist reviews and accepts -> influencer sees it.
"""
import json
import sys
import time

import httpx

from reset_demo import reset
from seed_demo import BASE, PASSWORD

AI = "http://127.0.0.1:8000"
BRIEF = "GlowFuel is launching a vegan protein bar and needs fitness creators to make short videos showing it in a real workout routine."
ANSWER = "Budget is 50k and applications close on 15 December."


def login(email: str) -> dict[str, str]:
    response = httpx.post(f"{BASE}/api/auth/login", json={"email": email, "password": PASSWORD}, timeout=30)
    response.raise_for_status()
    return {"Authorization": f"Bearer {response.json()['token']}"}


def stream(path: str, headers: dict[str, str], body: dict) -> tuple[str, dict]:
    """Returns the run id and the final event, which is a pause or the finished result."""
    run_id, last = "", {}
    with httpx.stream("POST", f"{AI}{path}", json=body, headers=headers, timeout=300, trust_env=False) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            event = json.loads(line)
            if event["type"] == "step":
                print(f"    {event['message']}", flush=True)
            elif event["type"] == "run":
                run_id = event["run_id"]
            else:
                last = event
    if last.get("type") == "error":
        raise SystemExit(f"Agent error: {last['message']}")
    return run_id, last


def stage(name: str, started: float) -> float:
    now = time.perf_counter()
    print(f"  [{name}: {now - started:.0f}s]\n", flush=True)
    return now


def main() -> None:
    print("Resetting demo data")
    reset()
    begin = clock = time.perf_counter()

    brand = login("brand@demo.test")
    print("1. Campaign Copilot")
    run, event = stream("/agents/copilot/runs", brand, {"brief": BRIEF})
    assert event["payload"]["kind"] == "question", event
    print("    Asked:", event["payload"]["questions"])
    _, event = stream(f"/agents/runs/{run}/resume", brand, {"value": {"answer": ANSWER}})
    assert event["payload"]["kind"] == "approve_campaign", event
    proposal = event["payload"]["proposal"]
    print("    Proposal:", proposal)
    assert proposal["budget"] == "50000" and proposal["brandName"] == "GlowFuel"
    _, event = stream(f"/agents/runs/{run}/resume", brand, {"value": {"approved": True, "campaign": proposal}})
    assert event["result"]["status"] == "DONE", event
    clock = stage("Copilot", clock)

    meera = login("meera@demo.test")
    print("2. Opportunity Scout")
    run, event = stream("/agents/scout/runs", meera, {})
    assert event["payload"]["kind"] == "choose_campaign", event
    matches = event["payload"]["matches"]
    for match in matches:
        print(f"    {match['fit']}/5 {match['campaign']['title']}: {match['reason']} (quote: {match['profile_quote']})")
    chosen = next((m for m in matches if m["campaign"]["title"] == proposal["title"]), None)
    assert chosen, f"The Scout did not offer the new campaign: {proposal['title']}"
    campaign_id = chosen["campaign"]["id"]
    _, event = stream(f"/agents/runs/{run}/resume", meera, {"value": {"campaign_id": campaign_id}})
    assert event["payload"]["kind"] == "approve_application", event
    draft = event["payload"]["draft"]
    print("    Draft:", draft)
    application = {k: draft[k] for k in ("message", "influencerName", "niche", "platform")}
    _, event = stream(f"/agents/runs/{run}/resume", meera, {"value": {"approved": True, "application": application}})
    assert event["result"]["status"] == "DONE", event
    clock = stage("Scout", clock)

    print("3. Applicant Shortlist")
    run, event = stream("/agents/shortlist/runs", brand, {"campaign_id": campaign_id})
    assert event["payload"]["kind"] == "approve_shortlist", event
    evaluation = event["payload"]["evaluations"][0]
    print(
        f"    {evaluation['applicant_name']}: {evaluation['score']}/100 {evaluation['label']} "
        f"-> {evaluation['proposed_action']}; quotes: {evaluation['evidence_quotes']}"
    )
    decision = {"application_id": evaluation["application_id"], "action": "ACCEPT"}
    _, event = stream(f"/agents/runs/{run}/resume", brand, {"value": {"decisions": [decision]}})
    assert event["result"]["outcomes"][0]["outcome"] == "ACCEPTED", event
    clock = stage("Shortlist", clock)

    print("4. The existing influencer dashboard")
    dashboard = httpx.get(f"{BASE}/api/influencers/mydashboard", headers=meera, timeout=30).json()
    accepted = httpx.get(f"{BASE}/api/influencers/accepted", headers=meera, timeout=30).json()
    print("    Dashboard:", dashboard)
    print("    Accepted:", [c["title"] for c in accepted])
    assert dashboard["acceptedApplications"] == 1 and accepted[0]["id"] == campaign_id

    print(f"\nPASSED in {time.perf_counter() - begin:.0f}s")
    print("Resetting demo data")
    reset()


if __name__ == "__main__":
    sys.exit(main())
