"""Go/no-go test: assesses the demo applicants with the real local model. Reads only; changes nothing."""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.agents.shortlist import evaluate_application  # noqa: E402
from app.backend import BackendClient  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm import OllamaLLM  # noqa: E402
from seed_demo import ENV  # noqa: E402


async def main() -> None:
    login = httpx.post(
        f"{settings.backend_url}/api/auth/login",
        json={"email": "brand@demo.test", "password": ENV["DEMO_PASSWORD"]},
        timeout=30,
    )
    login.raise_for_status()
    backend = BackendClient(login.json()["token"])
    llm = OllamaLLM()
    campaign = await backend.get_campaign(1)
    applications = await backend.get_applications(1)
    print(f"model: {settings.ollama_model}; applications: {len(applications)}", flush=True)
    for app in applications:
        started = time.perf_counter()
        evaluation = await evaluate_application(llm, campaign, app)
        elapsed = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "name": evaluation["applicant_name"],
                    "seconds": round(elapsed, 1),
                    "status": evaluation["status"],
                    "score": evaluation["score"],
                    "label": evaluation["label"],
                    "action": evaluation["proposed_action"],
                    "quotes": evaluation["evidence_quotes"],
                    "flags": evaluation["flags"],
                    "strengths": evaluation["strengths"],
                    "concerns": evaluation["concerns"],
                },
                indent=1,
            ),
            flush=True,
        )
    await backend.aclose()


asyncio.run(main())
