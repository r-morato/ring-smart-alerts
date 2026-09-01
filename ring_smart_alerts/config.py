"""Configuration loading for ring-smart-alerts.

All runtime configuration comes from environment variables, optionally seeded
from a local ``.env`` file (see ``.env.example``). Nothing here reaches the
network or touches Ring / Home Assistant -- it only assembles a validated
:class:`Settings` object for the rest of the app.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional dependency -- only needed if a .env file is used
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is in requirements but keep it soft
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


#: Environment variables that must be present for the app to start.
REQUIRED_VARS = ("RING_EMAIL", "RING_PASSWORD", "HA_URL", "HA_TOKEN", "HA_NOTIFY_TARGET")


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


@dataclass(slots=True)
class Settings:
    """Validated runtime settings."""

    ring_email: str
    ring_password: str
    ha_url: str
    ha_token: str
    ha_notify_target: str

    min_confidence: float = 0.35
    event_kinds: frozenset[str] = frozenset({"motion", "ding"})
    notify_on_empty: bool = True

    token_cache_path: Path = field(
        default_factory=lambda: Path.home() / ".config" / "ring-smart-alerts" / "token.json"
    )
    snapshot_dir: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir()) / "ring-smart-alerts"
    )

    @classmethod
    def load(cls, env_file: str | os.PathLike[str] | None = ".env") -> "Settings":
        """Build :class:`Settings` from the environment.

        If *env_file* exists it is loaded first (without overriding variables
        that are already set in the real environment).
        """
        if env_file and Path(env_file).is_file():
            load_dotenv(env_file, override=False)

        missing = [name for name in REQUIRED_VARS if not os.environ.get(name)]
        if missing:
            raise ConfigError(
                "Missing required environment variable(s): "
                + ", ".join(missing)
                + ".\nCopy .env.example to .env and fill it in, or export them."
            )

        kinds = os.environ.get("EVENT_KINDS")
        event_kinds = (
            frozenset(k.strip().lower() for k in kinds.split(",") if k.strip())
            if kinds
            else cls.event_kinds
        )

        settings = cls(
            ring_email=os.environ["RING_EMAIL"],
            ring_password=os.environ["RING_PASSWORD"],
            ha_url=os.environ["HA_URL"].rstrip("/"),
            ha_token=os.environ["HA_TOKEN"],
            ha_notify_target=os.environ["HA_NOTIFY_TARGET"],
            min_confidence=_get_float("MIN_CONFIDENCE", 0.35),
            event_kinds=event_kinds,
            notify_on_empty=_get_bool("NOTIFY_ON_EMPTY", True),
        )

        if token_path := os.environ.get("TOKEN_CACHE_PATH"):
            settings.token_cache_path = Path(token_path).expanduser()
        if snap_dir := os.environ.get("SNAPSHOT_DIR"):
            settings.snapshot_dir = Path(snap_dir).expanduser()

        return settings


def _get_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
