import asyncio
from types import SimpleNamespace

from ring_smart_alerts.config import Settings
from ring_smart_alerts.detector import Detection
from ring_smart_alerts.main import _process_event


def make_settings(**overrides):
    return Settings(
        ring_email="a@b.c",
        ring_password="pw",
        ha_url="http://ha.local:8123",
        ha_token="tok123",
        ha_notify_target="mobile_app_pixel",
        **overrides,
    )


class FakeClient:
    def __init__(self, image):
        self.image = image

    async def get_snapshot(self, device):
        return self.image


class FakeDetector:
    def __init__(self, detections):
        self.detections = detections

    def detect(self, image):
        return self.detections


class FakeNotifier:
    def __init__(self):
        self.sent = []

    def notify(self, message, *, title, image):
        self.sent.append(message)


def run(image, detections, *, kind="ding", **settings):
    notifier = FakeNotifier()
    asyncio.run(
        _process_event(
            SimpleNamespace(name="Front Door"),
            SimpleNamespace(kind=kind),
            client=FakeClient(image),
            detector=FakeDetector(detections),
            notifier=notifier,
            settings=make_settings(**settings),
        )
    )
    return notifier.sent


def test_detection_is_notified():
    sent = run(b"jpeg", [Detection("person", 0.9)], notify_on_empty=False)
    assert len(sent) == 1
    assert "a person" in sent[0]


def test_empty_frame_suppressed_when_notify_on_empty_off():
    assert run(b"jpeg", [], kind="motion", notify_on_empty=False) == []


def test_missing_snapshot_still_alerts_when_notify_on_empty_off():
    sent = run(None, [], notify_on_empty=False)
    assert len(sent) == 1
    assert "no snapshot available" in sent[0]
