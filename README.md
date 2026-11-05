# InfluencerHub AI

A local demo layer that adds three agents to InfluencerHub, using Python, LangGraph, Pydantic and a local Qwen model through Ollama. It needs no Spring Boot changes and no paid LLM API.

| Agent | User | What it does | What needs approval |
|---|---|---|---|
| Applicant Shortlist | Brand | Ranks pending applicants with quoted evidence and proposes decisions | Each accept or reject |
| Campaign Copilot | Brand | Turns a rough brief into a campaign, asking about anything missing | Creating the campaign |
| Opportunity Scout | Influencer | Finds suitable campaigns and drafts a pitch | Submitting the application |

## How it works

```text
Angular -> FastAPI + LangGraph -> existing Spring Boot API -> PostgreSQL
                  |
                  +-> local Qwen through Ollama
```

- The agents act with the logged-in user's token, so Spring Boot still enforces roles and ownership. The token never reaches the model or saved graph state.
- The service has no database access and no cloud model fallback.
- The model only returns validated data and has no tools. After an approval, plain code calls the API with exactly what was approved.
- Text from users is treated as data. Quotes are checked against the source text, invented metrics are removed, and injection attempts are flagged and never proposed for acceptance.
- The score rubric is rubric v1 from `docs/IMPLEMENTATION_DESIGN.md`, which is still provisional.

## Run it

You need Docker, Java 17, Node 20, Python 3.12 and Ollama with the model pulled (`ollama pull qwen2.5:7b`).

Run these from `influencerHub-ai/`:

```bash
.venv/Scripts/python.exe scripts/make_env.py
```

```bash
docker compose -f docker-compose.demo.yml --env-file .env up -d --wait
```

Then start each of these in its own terminal:

```bash
powershell -ExecutionPolicy Bypass -File scripts/run-backend.ps1
```

```bash
powershell -ExecutionPolicy Bypass -File scripts/run-ai.ps1
```

```bash
powershell -ExecutionPolicy Bypass -File scripts/run-frontend.ps1
```

Once the backend is running, load the demo data:

```bash
.venv/Scripts/python.exe scripts/seed_demo.py
```

Open http://localhost:4200. The demo accounts are `brand@demo.test` and `meera@demo.test`, and their password is `DEMO_PASSWORD` in `.env`.

## Test it

```bash
.venv/Scripts/python.exe -m pytest -q tests
```

```bash
.venv/Scripts/python.exe scripts/model_gate.py
```

The second command runs the real model on the demo applicants and changes nothing.

To run the whole story through the live services, then reset the demo data:

```bash
.venv/Scripts/python.exe scripts/rehearse_demo.py
```

To reset the demo data on its own:

```bash
.venv/Scripts/python.exe scripts/reset_demo.py
```

## Demo script

Reset the data first and start with the model loaded, because the first call is slower.

1. **Brand, Campaign Copilot:** log in as `brand@demo.test`, open **AI Agents > Campaign Copilot** and enter the brief below. It asks for the budget and deadline. Answer "Budget is 50k and applications close on 15 December", review the card and approve it.

   > GlowFuel is launching a vegan protein bar and needs fitness creators to make short videos showing it in a real workout routine.

2. **Brand, Applicant Shortlist:** review **GlowFuel Plant Protein Launch**. Show the quoted evidence, the limits it discloses, and Vikram's blocked injection attempt. Approve the decisions.
3. **Influencer, Scout:** log in as `meera@demo.test`, open **AI Scout** and choose the new protein bar campaign. Edit the pitch if you like, then approve it.
4. **Brand, close the loop:** review the protein bar campaign and accept Meera.
5. **Influencer, the proof:** open **Accepted** and the dashboard, which are existing pages showing the real change.

Point out that the model runs on this laptop, nothing changes without approval, and Spring Boot still enforces permissions.

## Limits

- It only works locally, because the deployed Vercel and Render apps can't reach a model on a laptop. The production build shows that agents are unavailable.
- Paused runs are held in memory and are lost if the service restarts.
- Outreach drafts can be edited and copied, but there's no messaging feature to send them.
