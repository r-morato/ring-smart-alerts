import asyncio
from types import SimpleNamespace

import pytest

from ring_smart_alerts.config import ConfigError, Settings
from ring_smart_alerts.cooldown import MotionCooldown
from ring_smart_alerts.detector import Detection
from ring_smart_alerts.main import _process_event
from ring_smart_alerts.notifier import NotifyError


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_off_by_default():
    cd = MotionCooldown(0)
    cd.record("front", {"car"})
    assert not cd.is_repeat("front", {"car"})


def test_same_thing_is_quiet_new_thing_alerts():
    clock = Clock()
    cd = MotionCooldown(120, clock=clock)
    assert not cd.is_repeat("front", {"car"})
    cd.record("front", {"car"})
    clock.now += 30
    assert cd.is_repeat("front", {"car"})
    assert cd.is_repeat("front", set())  # nothing recognised adds nothing
    assert not cd.is_repeat("front", {"car", "person"})
    assert not cd.is_repeat("garden", {"car"})  # per camera


def test_window_expires_and_remembers_everything_seen():
    clock = Clock()
    cd = MotionCooldown(120, clock=clock)
    cd.record("front", {"car"})
    clock.now += 60
    cd.record("front", {"person"})  # new thing alerted within the window
    clock.now += 60
    assert cd.is_repeat("front", {"car"})  # window restarted at the person alert
    assert cd.is_repeat("front", {"person", "car"})
    clock.now += 61
    assert not cd.is_repeat("front", {"car"})


def test_config(monkeypatch):
    for k, v in {"RING_EMAIL": "a", "RING_PASSWORD": "b", "HA_URL": "http://ha",
                 "HA_TOKEN": "t", "HA_NOTIFY_TARGET": "x"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("MOTION_COOLDOWN", raising=False)
    assert Settings.load(env_file=None).motion_cooldown == 0
    monkeypatch.setenv("MOTION_COOLDOWN", "90")
    assert Settings.load(env_file=None).motion_cooldown == 90
    for bad in ("-5", "soon"):
        monkeypatch.setenv("MOTION_COOLDOWN", bad)
        with pytest.raises(ConfigError):
            Settings.load(env_file=None)


# --- through the real event handler -------------------------------------------------

class FakeClient:
    async def get_snapshot(self, device):
        return b"jpeg"


class ScriptedDetector:
    """Returns the next list of labels each time it is called."""

    def __init__(self, frames):
        self.frames = list(frames)

    def detect(self, image):
        return [Detection(label, 0.9) for label in self.frames.pop(0)]


class FakeNotifier:
    def __init__(self, fail_first=False):
        self.sent = []
        self.fail_first = fail_first

    def notify(self, message, *, title, image):
        if self.fail_first:
            self.fail_first = False
            raise NotifyError("HA down")
        self.sent.append(message)


def run_events(events, frames, cooldown, notifier=None):
    """events: list of (camera, kind). Returns the messages that were sent."""
    notifier = notifier or FakeNotifier()
    detector = ScriptedDetector(frames)
    settings = Settings(ring_email="a", ring_password="b", ha_url="http://ha", ha_token="t",
                        ha_notify_target="x", notify_on_empty=True)

    async def go():
        for camera, kind in events:
            await _process_event(
                SimpleNamespace(id=camera, name=camera.title()), SimpleNamespace(kind=kind),
                client=FakeClient(), detector=detector, notifier=notifier,
                settings=settings, cooldown=cooldown,
            )

    asyncio.run(go())
    return notifier.sent


def test_busy_camera_sends_one_alert_until_something_new_appears():
    clock = Clock()
    sent = run_events(
        [("front", "motion")] * 4 + [("garden", "motion")],
        [["car"], ["car"], [], ["car", "person"], ["car"]],
        MotionCooldown(120, clock=clock),
    )
    assert len(sent) == 3
    assert "a car" in sent[0]
    assert "a person" in sent[1]
    assert sent[2].startswith("Garden")


def test_doorbell_always_gets_through():
    sent = run_events(
        [("front", "motion"), ("front", "ding"), ("front", "ding")],
        [["person"], ["person"], ["person"]],
        MotionCooldown(120, clock=Clock()),
    )
    assert len(sent) == 3


def test_failed_notification_does_not_start_the_cooldown():
    notifier = FakeNotifier(fail_first=True)
    sent = run_events(
        [("front", "motion"), ("front", "motion")],
        [["car"], ["car"]],
        MotionCooldown(120, clock=Clock()),
        notifier=notifier,
    )
    assert len(sent) == 1  # the retry-worthy second event was not swallowed


def test_without_cooldown_every_event_alerts():
    sent = run_events([("front", "motion")] * 3, [["car"]] * 3, None)
    assert len(sent) == 3
