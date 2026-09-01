"""Send notifications to Home Assistant via its REST API.

Calls the ``notify.<target>`` service:
``POST {HA_URL}/api/services/notify/{target}`` with a bearer token.
"""

from __future__ import annotations

import logging

import requests

from .config import Settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15


class NotifyError(RuntimeError):
    """Raised when Home Assistant rejects the notification."""


class HANotifier:
    """Thin wrapper around the Home Assistant ``notify`` service."""

    def __init__(self, settings: Settings, *, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._url = f"{settings.ha_url}/api/services/notify/{settings.ha_notify_target}"
        self._headers = {
            "Authorization": f"Bearer {settings.ha_token}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout

    def notify(
        self,
        message: str,
        *,
        title: str = "Ring",
        image_path: str | None = None,
    ) -> None:
        """POST a notification. Raises :class:`NotifyError` on a non-2xx response.

        *image_path* is passed through as ``data.image``; whether HA uses it
        depends on the notify integration (the mobile app companion does).
        """
        payload: dict[str, object] = {"message": message, "title": title}
        if image_path:
            payload["data"] = {"image": image_path}

        try:
            resp = requests.post(
                self._url, json=payload, headers=self._headers, timeout=self._timeout
            )
        except requests.RequestException as exc:
            raise NotifyError(f"Could not reach Home Assistant at {self._url}: {exc}") from exc

        if not resp.ok:
            body = resp.text[:500]
            logger.error("HA notify failed: %s %s -- %s", resp.status_code, resp.reason, body)
            raise NotifyError(f"Home Assistant returned {resp.status_code}: {body}")

        logger.debug("HA notify ok (%s)", resp.status_code)
