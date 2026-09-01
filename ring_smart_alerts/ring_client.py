"""Ring authentication, snapshot fetching, and event listening.

Wraps the async interface of ``ring-doorbell`` (>= 0.9.14). The library's API is
async-first and has shifted across versions, so everything here goes through the
``async_*`` methods.

Auth token and the Firebase Cloud Messaging (FCM) credentials used by the event
listener are both cached to a single JSON file so that 2FA is only needed on the
very first run.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from ring_doorbell import Auth, AuthenticationError, Requires2FAError, Ring, RingError
from ring_doorbell.listen import RingEventListener

from .config import Settings

logger = logging.getLogger(__name__)

USER_AGENT = "ring-smart-alerts/0.1"

#: Ring endpoint for the last stored snapshot image (see ring_doorbell.const)
_SNAPSHOT_IMAGE_ENDPOINT = "/clients_api/snapshots/image/{0}"

class RingAuthError(RuntimeError):
    """Ring authentication failed for a reason the user needs to act on."""


#: async callback invoked for each accepted event: (device, event) -> None
EventHandler = Callable[[Any, Any], Awaitable[None]]
#: prompt for the 2FA code; overridable in tests
OtpProvider = Callable[[], str]


def _default_otp() -> str:
    return input("Ring 2FA code (sent by email/SMS): ").strip()


class RingClient:
    """Connect to Ring, pull snapshots, and dispatch motion/ding events."""

    def __init__(self, settings: Settings, *, otp_provider: OtpProvider = _default_otp) -> None:
        self._settings = settings
        self._otp_provider = otp_provider
        self._cache_path = settings.token_cache_path
        self._ring: Ring | None = None
        self._auth: Auth | None = None
        self._listener: RingEventListener | None = None

    # ------------------------------------------------------------------ auth

    def _read_cache(self) -> dict[str, Any]:
        if self._cache_path.is_file():
            try:
                return json.loads(self._cache_path.read_text())
            except (json.JSONDecodeError, OSError):
                logger.warning("Ignoring unreadable token cache at %s", self._cache_path)
        return {}

    def _write_cache(self, **updates: Any) -> None:
        data = self._read_cache()
        data.update(updates)
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(json.dumps(data))
        # token/credentials are secrets -- keep them owner-only where supported
        with contextlib.suppress(OSError):  # e.g. Windows without full ACL support
            self._cache_path.chmod(0o600)

    def _on_token_updated(self, token: dict[str, Any]) -> None:
        self._write_cache(token=token)

    def _on_credentials_updated(self, credentials: dict[str, Any]) -> None:
        self._write_cache(fcm=credentials)

    async def _do_2fa_auth(self) -> Auth:
        s = self._settings
        auth = Auth(USER_AGENT, None, self._on_token_updated)
        try:
            try:
                await auth.async_fetch_token(s.ring_email, s.ring_password)
            except Requires2FAError:
                code = self._otp_provider()
                try:
                    await auth.async_fetch_token(s.ring_email, s.ring_password, code)
                except AuthenticationError as exc:
                    raise RingAuthError(
                        f"Ring rejected the 2FA code: {exc}. It may have expired -- try again."
                    ) from exc
            except AuthenticationError as exc:
                raise RingAuthError(
                    "Ring rejected RING_EMAIL / RING_PASSWORD "
                    f"({exc}). Check them in .env; note repeated failures can trigger a "
                    "temporary lockout."
                ) from exc
        except BaseException:
            await auth.async_close()
            raise
        return auth

    async def connect(self) -> None:
        """Establish an authenticated session, reusing the cached token if valid."""
        cache = self._read_cache()
        token = cache.get("token")

        if token:
            self._auth = Auth(USER_AGENT, token, self._on_token_updated)
            self._ring = Ring(self._auth)
            try:
                await self._ring.async_create_session()
            except AuthenticationError:
                logger.info("Cached Ring token expired -- re-authenticating")
                self._auth = await self._do_2fa_auth()
                self._ring = Ring(self._auth)
        else:
            self._auth = await self._do_2fa_auth()
            self._ring = Ring(self._auth)

        await self._ring.async_update_data()
        count = len(self._ring.devices().all_devices)
        logger.info("Connected to Ring; %d device(s) found", count)

    # -------------------------------------------------------------- snapshots

    async def get_snapshot(self, device: Any) -> bytes | None:
        """Return JPEG bytes for *device*, or ``None`` if no image is available.

        Battery/low-power cameras cannot produce a *fresh* snapshot while they
        are recording a motion clip, and without a Ring subscription there is no
        snapshot-on-motion. We therefore:

        1. ask for a fresh snapshot, polling generously (~16s);
        2. fall back to the last stored snapshot Ring has (may be stale);
        3. give up and return ``None`` so the caller can still send a text alert.
        """
        try:
            data = await device.async_get_snapshot(retries=8, delay=2)
            if data:
                return data
        except Exception as exc:  # noqa: BLE001 - library raises a grab-bag of errors
            logger.debug("Fresh snapshot failed for %s: %s", device.name, exc)

        # Fall back to whatever image Ring already has on file.
        try:
            resp = await self._ring.async_query(_SNAPSHOT_IMAGE_ENDPOINT.format(device.id))
            if resp.content:
                logger.info("Using last stored snapshot for %s (no fresh one)", device.name)
                return resp.content
        except Exception as exc:  # noqa: BLE001
            logger.debug("Stored snapshot fallback failed for %s: %s", device.name, exc)

        logger.warning("No snapshot available for %s", device.name)
        return None

    # ----------------------------------------------------------------- events

    def _find_device(self, doorbot_id: int) -> Any | None:
        assert self._ring is not None
        try:
            return self._ring.devices().get_video_device(doorbot_id)
        except RingError:
            return None

    async def listen(self, on_event: EventHandler, *, stop: asyncio.Event | None = None) -> None:
        """Start the FCM listener and dispatch accepted events until *stop* is set."""
        assert self._ring is not None
        cache = self._read_cache()
        loop = asyncio.get_running_loop()
        stop = stop or asyncio.Event()

        self._listener = RingEventListener(
            self._ring, cache.get("fcm"), self._on_credentials_updated
        )

        def _dispatch(event: Any) -> None:
            # Called synchronously by firebase-messaging; hop back onto the loop.
            asyncio.run_coroutine_threadsafe(self._handle(event, on_event), loop)

        self._listener.add_notification_callback(_dispatch)

        if not await self._listener.start():
            raise RuntimeError("Ring event listener failed to start")
        logger.info("Listening for Ring events (%s)", ", ".join(sorted(self._settings.event_kinds)))

        try:
            await stop.wait()
        finally:
            await self._listener.stop()

    async def _handle(self, event: Any, on_event: EventHandler) -> None:
        if event.is_update or event.kind not in self._settings.event_kinds:
            logger.debug("Skipping event kind=%s is_update=%s", event.kind, event.is_update)
            return

        device = self._find_device(event.doorbot_id)
        if device is None:
            logger.warning(
                "Event for unknown device id=%s (%s)", event.doorbot_id, event.device_name
            )
            return

        try:
            await on_event(device, event)
        except Exception:  # noqa: BLE001 - never let one bad event kill the listener
            logger.exception("Handler failed for %s event on %s", event.kind, event.device_name)

    async def close(self) -> None:
        if self._auth is not None:
            await self._auth.async_close()
