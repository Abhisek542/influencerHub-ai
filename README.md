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
