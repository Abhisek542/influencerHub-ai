"""Calls the existing Spring Boot API as the logged-in user, so it keeps enforcing roles and ownership.

The token lives only in this client. It is never placed in graph state or sent to the model.
"""
import httpx

from .config import settings


class BackendError(Exception):
    def __init__(self, status: int, code: str):
        super().__init__(f"{code} ({status})")
        self.status = status
        self.code = code


class BackendClient:
    def __init__(self, token: str, transport: httpx.AsyncBaseTransport | None = None):
        self._client = httpx.AsyncClient(
            base_url=settings.backend_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
            transport=transport,
            trust_env=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _request(self, method: str, path: str, **kwargs) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as error:
            raise BackendError(503, "backend_unavailable") from error
        if response.status_code == 401:
            raise BackendError(401, "session_expired")
        if response.status_code == 403:
            raise BackendError(403, "forbidden")
        if response.status_code == 404:
            raise BackendError(404, "not_found")
        if response.status_code >= 400:
            raise BackendError(response.status_code, "backend_error")
        return response

    # Brand
    async def get_campaign(self, campaign_id: int) -> dict:
        return (await self._request("GET", f"/api/campaign/{campaign_id}")).json()

    async def get_applications(self, campaign_id: int) -> list[dict]:
        return (await self._request("GET", f"/api/brand/applications/{campaign_id}")).json()

    async def set_application_status(self, application_id: int, status: str) -> None:
        await self._request("POST", f"/api/brand/applications/{application_id}/status", params={"status": status})

    async def create_campaign(self, campaign: dict) -> None:
        await self._request("POST", "/api/campaign/create", json=campaign)

    # Influencer
    async def get_profile(self) -> dict:
        return (await self._request("GET", "/api/influencer/profile/me")).json()

    async def browse_campaigns(self, size: int = 50) -> list[dict]:
        return (await self._request("GET", "/api/campaign/filtr", params={"size": size})).json()["content"]

    async def apply(self, campaign_id: int, application: dict) -> str:
        response = await self._request("POST", f"/api/influencer/apply/{campaign_id}", json=application)
        return response.json().get("message", "")
