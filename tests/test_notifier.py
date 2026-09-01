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
    def __init__(self, status_code=200, text="ok"):
        self.status_code = status_code
        self.reason = "OK" if status_code < 400 else "Bad"
        self.text = text

    @property
    def ok(self):
        return self.status_code < 400


def test_notify_posts_expected_payload(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr("ring_smart_alerts.notifier.requests.post", fake_post)

    HANotifier(make_settings()).notify("a person", title="Ring", image_path="/tmp/x.jpg")

    assert captured["url"] == "http://ha.local:8123/api/services/notify/mobile_app_pixel"
    assert captured["headers"]["Authorization"] == "Bearer tok123"
    assert captured["json"]["message"] == "a person"
    assert captured["json"]["title"] == "Ring"
    assert captured["json"]["data"] == {"image": "/tmp/x.jpg"}


def test_notify_omits_data_without_image(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "ring_smart_alerts.notifier.requests.post",
        lambda url, json, headers, timeout: captured.update(json=json) or FakeResponse(),
    )
    HANotifier(make_settings()).notify("motion")
    assert "data" not in captured["json"]


def test_notify_raises_on_error(monkeypatch):
    monkeypatch.setattr(
        "ring_smart_alerts.notifier.requests.post",
        lambda *a, **k: FakeResponse(status_code=401, text="unauthorized"),
    )
    with pytest.raises(NotifyError):
        HANotifier(make_settings()).notify("x")
