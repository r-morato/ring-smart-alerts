"""Send notifications to Home Assistant via its REST API.

Calls the ``notify.<target>`` service:
``POST {HA_URL}/api/services/notify/{target}`` with a bearer token.

When a snapshot is supplied it is first uploaded to Home Assistant's own image
store (``POST /api/image/upload``) and the notification references it as
``/api/image/serve/<id>/original`` -- so the companion app fetches the picture
straight from your HA instance, and the image never goes to a third party. Only
the last :data:`KEEP_IMAGES` uploads are retained; older ones are deleted over
HA's websocket API (the image store has no REST delete).
"""

from __future__ import annotations

import asyncio
import collections
import logging

import aiohttp
import requests

from .config import Settings

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 15

#: How many uploaded snapshots to keep in HA's image store before pruning.
KEEP_IMAGES = 5


class NotifyError(RuntimeError):
    """Raised when Home Assistant rejects the notification."""


class HANotifier:
    """Thin wrapper around the Home Assistant ``notify`` service."""

    def __init__(self, settings: Settings, *, timeout: int = DEFAULT_TIMEOUT) -> None:
        self._base = settings.ha_url
        self._url = f"{settings.ha_url}/api/services/notify/{settings.ha_notify_target}"
        self._token = settings.ha_token
        self._auth = {"Authorization": f"Bearer {settings.ha_token}"}
        self._timeout = timeout
        self._recent_images: collections.deque[str] = collections.deque()

    def notify(
        self,
        message: str,
        *,
        title: str = "Ring",
        image: bytes | None = None,
    ) -> None:
        """POST a notification. Raises :class:`NotifyError` on a non-2xx response.

        *image* (JPEG bytes) is uploaded to HA and attached; if the upload fails
        the alert is still sent, text-only.
        """
        payload: dict[str, object] = {"message": message, "title": title}

        if image:
            ref = self._upload_image(image)
            if ref:
                payload["data"] = {"image": ref}

        try:
            resp = requests.post(
                self._url,
                json=payload,
                headers={**self._auth, "Content-Type": "application/json"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise NotifyError(f"Could not reach Home Assistant at {self._url}: {exc}") from exc

        if not resp.ok:
            body = resp.text[:500]
            logger.error("HA notify failed: %s %s -- %s", resp.status_code, resp.reason, body)
            raise NotifyError(f"Home Assistant returned {resp.status_code}: {body}")

        logger.debug("HA notify ok (%s)", resp.status_code)

    # ----------------------------------------------------------------- images

    def _upload_image(self, image: bytes) -> str | None:
        """Upload *image* to HA's image store, returning its serve path or ``None``."""
        try:
            resp = requests.post(
                f"{self._base}/api/image/upload",
                headers=self._auth,
                files={"file": ("snapshot.jpg", image, "image/jpeg")},
                timeout=self._timeout,
            )
            resp.raise_for_status()
            image_id = resp.json()["id"]
        except (requests.RequestException, ValueError, KeyError) as exc:
            logger.warning("Snapshot upload to HA failed (%s); sending text-only alert", exc)
            return None

        self._recent_images.append(image_id)
        self._prune_images()
        return f"/api/image/serve/{image_id}/original"

    def _prune_images(self) -> None:
        while len(self._recent_images) > KEEP_IMAGES:
            old = self._recent_images.popleft()
            try:
                asyncio.run(self._ws_delete(old))
            except Exception as exc:  # noqa: BLE001 - pruning is best-effort
                logger.debug("Could not delete old HA image %s: %s", old, exc)

    async def _ws_delete(self, image_id: str) -> None:
        ws_url = self._base.replace("http", "ws", 1) + "/api/websocket"

        async def _go() -> None:
            async with aiohttp.ClientSession() as session, session.ws_connect(ws_url) as ws:
                await ws.receive_json()  # auth_required
                await ws.send_json({"type": "auth", "access_token": self._token})
                await ws.receive_json()  # auth_ok
                await ws.send_json({"id": 1, "type": "image/delete", "image_id": image_id})
                await ws.receive_json()

        await asyncio.wait_for(_go(), timeout=self._timeout)
