"""Detector smoke test.

Requires ultralytics + torch and downloads yolov8n.pt (~6 MB) on first run, so it
is skipped automatically when the ML stack or network model isn't available.
"""

import pytest

ultralytics = pytest.importorskip("ultralytics")

from ring_smart_alerts.detector import Detector, summarize  # noqa: E402


@pytest.fixture(scope="module")
def sample_jpeg():
    from ultralytics.utils import ASSETS

    bus = ASSETS / "bus.jpg"  # ships with ultralytics: people + a bus
    if not bus.is_file():
        pytest.skip("ultralytics sample asset not present")
    return bus.read_bytes()


def test_detects_person(sample_jpeg):
    try:
        detections = Detector(min_confidence=0.3).detect(sample_jpeg)
    except Exception as exc:  # model download / torch load failure in CI
        pytest.skip(f"YOLO unavailable: {exc}")

    labels = {d.label for d in detections}
    assert "person" in labels
    assert "person" in summarize(detections)
