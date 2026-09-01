import pytest

from ring_smart_alerts.config import ConfigError, Settings

BASE_ENV = {
    "RING_EMAIL": "a@b.c",
    "RING_PASSWORD": "pw",
    "HA_URL": "http://ha.local:8123/",
    "HA_TOKEN": "tok",
    "HA_NOTIFY_TARGET": "mobile_app_pixel",
}


@pytest.fixture
def clean_env(monkeypatch):
    for key in (*BASE_ENV, "MIN_CONFIDENCE", "EVENT_KINDS", "NOTIFY_ON_EMPTY"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_load_ok(clean_env):
    for k, v in BASE_ENV.items():
        clean_env.setenv(k, v)
    clean_env.setenv("MIN_CONFIDENCE", "0.5")
    clean_env.setenv("EVENT_KINDS", "motion")

    s = Settings.load(env_file=None)

    assert s.ha_url == "http://ha.local:8123"  # trailing slash stripped
    assert s.min_confidence == 0.5
    assert s.event_kinds == frozenset({"motion"})


def test_missing_vars_listed(clean_env):
    clean_env.setenv("RING_EMAIL", "a@b.c")
    with pytest.raises(ConfigError) as exc:
        Settings.load(env_file=None)
    assert "RING_PASSWORD" in str(exc.value)
    assert "HA_TOKEN" in str(exc.value)


def test_bad_confidence(clean_env):
    for k, v in BASE_ENV.items():
        clean_env.setenv(k, v)
    clean_env.setenv("MIN_CONFIDENCE", "high")
    with pytest.raises(ConfigError):
        Settings.load(env_file=None)
