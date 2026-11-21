"""Per-camera cooldown for motion alerts.

A busy camera (a parked car in view, a tree in the wind, someone pottering
about the garden) fires a motion event every time Ring sees movement, and each
one used to become a notification. With a cooldown, once a camera has alerted,
further motion from that camera is skipped for ``MOTION_COOLDOWN`` seconds
*unless it shows something new*: a car followed by more car stays quiet, but a
person turning up after the car still alerts.

Comparison uses the coarse YOLO labels (``person``, ``car``, ``dog``...), not
the CLIP wording, so "an adult" then "an adult (looks like a woman)" counts as
the same thing. Doorbell rings never go through here: someone pressing the
button should always reach you.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(slots=True)
class _Recent:
    sent_at: float
    labels: frozenset[str]


class MotionCooldown:
    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds = seconds
        self._clock = clock
        self._recent: dict[str, _Recent] = {}

    def is_repeat(self, camera: str, labels: Iterable[str]) -> bool:
        """True if *camera* alerted recently and *labels* add nothing new."""
        if self.seconds <= 0:
            return False
        recent = self._recent.get(camera)
        if recent is None or self._clock() - recent.sent_at >= self.seconds:
            return False
        return frozenset(labels) <= recent.labels

    def record(self, camera: str, labels: Iterable[str]) -> None:
        """Note that an alert was sent; restarts the camera's cooldown window."""
        if self.seconds <= 0:
            return
        now = self._clock()
        labels = frozenset(labels)
        recent = self._recent.get(camera)
        if recent is not None and now - recent.sent_at < self.seconds:
            labels |= recent.labels  # still within the window: remember everything seen
        self._recent[camera] = _Recent(sent_at=now, labels=labels)
