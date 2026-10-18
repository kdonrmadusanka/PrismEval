from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


class EvaluationRepository:
    _token: Optional[str] = None
    _token_expires_at: float = 0.0

    @staticmethod
    def _base_url() -> str:
        # Use https://open.larksuite.com for the international Lark edition.
        return os.getenv("FEISHU_API_BASE", "https://open.feishu.cn").rstrip("/")

    @classmethod
    async def _get_tenant_token(cls, client: httpx.AsyncClient) -> Optional[str]:
        app_id = os.getenv("FEISHU_APP_ID")
        app_secret = os.getenv("FEISHU_APP_SECRET")

        if not (app_id and app_secret):
            # Fallback: a manually supplied token (expires after ~2 hours).
            return os.getenv("FEISHU_TENANT_ACCESS_TOKEN")

        if cls._token and time.monotonic() < cls._token_expires_at:
            return cls._token

        response = await client.post(
            f"{cls._base_url()}/open-apis/auth/v3/tenant_access_token/internal",
            json={"app_id": app_id, "app_secret": app_secret},
        )
        data = response.json()
        token = data.get("tenant_access_token")
        if response.status_code != 200 or data.get("code") != 0 or not token:
            logger.error(
                "Feishu token request failed: http=%s code=%s msg=%s",
                response.status_code, data.get("code"), data.get("msg"),
            )
            return None

        expires_in = int(data.get("expire", 7200))
        cls._token = token
        cls._token_expires_at = time.monotonic() + max(expires_in - 60, 0)
        return token

    @classmethod
    async def sync_to_feishu_bitable(
        cls, student_id: str, eqi: float, risk_flags: List[str]
    ) -> bool:
        """
        Push one evaluation to a Feishu Bitable table.
        Returns True on success and False on any failure; it never raises, so a
        Feishu outage cannot break the evaluation response.
        """
        # Local development: skip the network and report success.
        if os.getenv("FEISHU_MOCK", "").strip().lower() in {"1", "true", "yes"}:
            return True

        app_token = os.getenv("FEISHU_APP_TOKEN")
        table_id = os.getenv("FEISHU_TABLE_ID")
        if not app_token or not table_id:
            logger.warning("Feishu sync skipped: FEISHU_APP_TOKEN or FEISHU_TABLE_ID not set.")
            return False

        payload = {
            "fields": {
                "Student ID": student_id,
                "EQI Score": round(float(eqi), 2),
                "Risk Status": "Flagged" if risk_flags else "Cleared",
            }
        }
        url = f"{cls._base_url()}/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"

        try:
            timeout = httpx.Timeout(15.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                token = await cls._get_tenant_token(client)
                if not token:
                    logger.warning("Feishu sync skipped: no tenant access token available.")
                    return False

                response = await client.post(
                    url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json; charset=utf-8",
                    },
                )
                body = response.json()

            if response.status_code == 200 and body.get("code") == 0:
                return True

            # Drop the cached token so the next call fetches a fresh one.
            cls._token = None
            logger.error(
                "Feishu sync failed: http=%s code=%s msg=%s",
                response.status_code, body.get("code"), body.get("msg"),
            )
            return False

        except (httpx.HTTPError, ValueError) as exc:
            # Log the exception type only, never headers or the token.
            logger.error("Feishu sync error: %s", type(exc).__name__)
            return False