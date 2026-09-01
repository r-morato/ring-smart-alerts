import pytest

from ring_smart_alerts.config import Settings
from ring_smart_alerts.notifier import HANotifier, NotifyError


def make_settings():
    return Settings(
        ring_email="a@b.c",
        ring_password="pw",
        ha_url="http://ha.local:8123",
        ha_token="tok123",
        ha_notify_target="mobile_app_pixel",
    )


class FakeResponse:
    def __init__(self, status_code=200, text="ok", payload=None):
        self.status_code = status_code
        self.reason = "OK" if status_code < 400 else "Bad"
        self.text = text
        self._payload = payload or {}

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._payload

    def raise_for_status(self):
        if not self.ok:
            raise _RequestsError(f"HTTP {self.status_code}")


# stand-in for requests.RequestException
import requests  # noqa: E402

_RequestsError = requests.RequestException


@pytest.fixture
def posts(monkeypatch):
    """Capture every requests.post call; drive responses by URL suffix."""
    calls = []
    responses = {}

    def fake_post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        for suffix, resp in responses.items():
            if url.endswith(suffix):
                if isinstance(resp, Exception):
                    raise resp
                return resp
        return FakeResponse()

    monkeypatch.setattr("ring_smart_alerts.notifier.requests.post", fake_post)
    return calls, responses


def test_text_only_notification(posts):
    calls, _ = posts
    HANotifier(make_settings()).notify("motion", title="Ring")

    assert len(calls) == 1
    assert calls[0]["url"] == "http://ha.local:8123/api/services/notify/mobile_app_pixel"
    assert calls[0]["headers"]["Authorization"] == "Bearer tok123"
    assert calls[0]["json"] == {"message": "motion", "title": "Ring"}


def test_image_is_uploaded_then_referenced(posts):
    calls, responses = posts
    responses["/api/image/upload"] = FakeResponse(payload={"id": "abc123"})

    HANotifier(make_settings()).notify("a person", image=b"\xff\xd8jpeg")

    upload, notify = calls
    assert upload["url"] == "http://ha.local:8123/api/image/upload"
    assert upload["files"]["file"][1] == b"\xff\xd8jpeg"
    assert notify["json"]["data"] == {"image": "/api/image/serve/abc123/original"}


def test_upload_failure_falls_back_to_text(posts):
    calls, responses = posts
    responses["/api/image/upload"] = _RequestsError("boom")

    HANotifier(make_settings()).notify("a person", image=b"jpeg")

    assert "data" not in calls[-1]["json"]
    assert calls[-1]["url"].endswith("/notify/mobile_app_pixel")


def test_notify_raises_on_error(posts):
    _, responses = posts
    responses["/notify/mobile_app_pixel"] = FakeResponse(status_code=401, text="unauthorized")

    with pytest.raises(NotifyError):
        HANotifier(make_settings()).notify("x")


def test_old_images_are_pruned(monkeypatch, posts):
    _, responses = posts
    ids = iter(f"id{n}" for n in range(100))
    responses["/api/image/upload"] = None  # replaced per-call below

    def fake_post(url, **kwargs):
        if url.endswith("/api/image/upload"):
            return FakeResponse(payload={"id": next(ids)})
        return FakeResponse()

    monkeypatch.setattr("ring_smart_alerts.notifier.requests.post", fake_post)

    deleted = []
    monkeypatch.setattr(HANotifier, "_ws_delete", _record(deleted))

    notifier = HANotifier(make_settings())
    for _ in range(8):
        notifier.notify("x", image=b"j")

    # keep the last 5, so the first 3 get pruned
    assert deleted == ["id0", "id1", "id2"]


def _record(sink):
    async def _fn(self, image_id):
        sink.append(image_id)

    return _fn
