"""Entry point: wire Ring events -> YOLO detection -> Home Assistant notification.

Run with::

    py -3.12 -m ring_smart_alerts.main
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal

from .config import ConfigError, Settings
from .detector import Detector, summarize
from .notifier import HANotifier, NotifyError
from .ring_client import RingAuthError, RingClient

logger = logging.getLogger(__name__)


async def _process_event(
    device,
    event,
    *,
    client: RingClient,
    detector: Detector,
    notifier: HANotifier,
    settings: Settings,
) -> None:
    try:
        image = await client.get_snapshot(device)

        if image is not None:
            detections = await asyncio.to_thread(detector.detect, image)
            phrase = summarize(detections)
        else:
            detections = []
            phrase = "no snapshot available"

        if not detections and not settings.notify_on_empty:
            logger.info("%s on %s: nothing recognised (suppressed)", event.kind, device.name)
            return

        verb = "rang the doorbell" if event.kind == "ding" else "motion"
        message = f"{device.name}: {verb} — {phrase}"
        logger.info(message)

        await asyncio.to_thread(
            notifier.notify,
            message,
            title=f"Ring – {device.name}",
            image=image,
        )
    except NotifyError:
        logger.exception("Failed to notify Home Assistant")


async def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        settings = Settings.load()
    except ConfigError as exc:
        raise SystemExit(str(exc)) from exc

    detector = Detector(
        min_confidence=settings.min_confidence,
        enable_clip=settings.enable_clip,
        clip_model=settings.clip_model,
        clip_pretrained=settings.clip_pretrained,
    )
    notifier = HANotifier(settings)
    client = RingClient(settings)

    try:
        await client.connect()
    except RingAuthError as exc:
        raise SystemExit(f"Ring authentication failed: {exc}") from exc

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):  # SIGTERM handler unsupported on Windows
            loop.add_signal_handler(sig, stop.set)

    async def handle(device, event) -> None:
        await _process_event(
            device, event,
            client=client, detector=detector, notifier=notifier, settings=settings,
        )

    try:
        await client.listen(handle, stop=stop)
    finally:
        await client.close()


def _run() -> None:
    """Sync entry point for the ``ring-smart-alerts`` console script."""
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())


if __name__ == "__main__":
    _run()
