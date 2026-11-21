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
from .cooldown import MotionCooldown
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
    cooldown: MotionCooldown | None = None,
) -> None:
    try:
        image = await client.get_snapshot(device)

        if image is not None:
            detections = await asyncio.to_thread(detector.detect, image)
            phrase = summarize(detections)
        else:
            detections = []
            phrase = "no snapshot available"

        # Only suppress a frame we actually looked at; a missing snapshot still alerts.
        if image is not None and not detections and not settings.notify_on_empty:
            logger.info("%s on %s: nothing recognised (suppressed)", event.kind, device.name)
            return

        # Rings always get through; repeated motion that shows nothing new doesn't.
        camera = str(getattr(device, "id", None) or device.name)
        labels = {d.label for d in detections}
        if cooldown and event.kind != "ding" and cooldown.is_repeat(camera, labels):
            logger.info("%s on %s: %s (cooldown, nothing new)", event.kind, device.name, phrase)
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
        if cooldown:
            cooldown.record(camera, labels)
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
    cooldown = MotionCooldown(settings.motion_cooldown)
    if settings.motion_cooldown > 0:
        logger.info("Motion cooldown: %ss per camera", f"{settings.motion_cooldown:g}")

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
            cooldown=cooldown,
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
