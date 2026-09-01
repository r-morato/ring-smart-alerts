"""Detector wiring tests with a fake YOLO model and a fake CLIP classifier.

These exercise ``Detector.detect`` -- per-person refinement and the whole-frame
fallback -- without pulling in torch/ultralytics/open-clip.
"""

import io

import pytest

from ring_smart_alerts.detector import Detection, Detector, summarize

pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


def _png(size=(40, 60)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, "PNG")
    return buf.getvalue()


class _Col(list):
    """Stands in for an ultralytics tensor column."""

    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, cls, conf, xyxy):
        self.cls = _Col(cls)
        self.conf = _Col(conf)
        self.xyxy = _Col(xyxy)


class _Result:
    names = {0: "person", 16: "dog", 2: "car"}

    def __init__(self, cls, conf, xyxy):
        self.boxes = _Boxes(cls, conf, xyxy)


def _fake_model(cls, conf, xyxy):
    result = _Result(cls, conf, xyxy)

    class _Model:
        def predict(self, *_a, **_k):
            return [result]

    return _Model()


def _detector(monkeypatch, *, model, classifier):
    det = Detector(enable_clip=True)
    monkeypatch.setattr(det, "_load", lambda: model)
    monkeypatch.setattr(det, "_classifier", lambda: classifier)
    return det


def test_person_box_is_refined(monkeypatch):
    calls = []

    class _Clf:
        def describe_person(self, _pil, box):
            calls.append(box)
            return "adult (looks like a man)"

        def classify_scene(self, _pil):  # pragma: no cover - not hit here
            raise AssertionError("scene fallback should not run when YOLO found a person")

    det = _detector(
        monkeypatch,
        model=_fake_model([0, 16], [0.9, 0.8], [[1, 1, 10, 30], [20, 20, 30, 40]]),
        classifier=_Clf(),
    )
    out = det.detect(_png())

    assert len(calls) == 1  # one person crop handed to CLIP
    assert summarize(out) == "an adult (looks like a man) and a dog"


def test_multiple_people_each_refined(monkeypatch):
    labels = iter(["child", "adult"])

    class _Clf:
        def describe_person(self, _pil, _box):
            return next(labels)

        def classify_scene(self, _pil):  # pragma: no cover
            return None

    det = _detector(
        monkeypatch,
        model=_fake_model([0, 0], [0.95, 0.6], [[0, 0, 5, 5], [6, 6, 12, 20]]),
        classifier=_Clf(),
    )
    out = det.detect(_png())
    # highest-confidence person first
    assert summarize(out) == "a child and an adult"


def test_low_confidence_boxes_dropped(monkeypatch):
    class _Clf:
        def describe_person(self, _pil, _box):  # pragma: no cover
            raise AssertionError("sub-threshold box should never reach CLIP")

        def classify_scene(self, _pil):
            return None

    det = _detector(
        monkeypatch,
        model=_fake_model([0], [0.10], [[0, 0, 5, 5]]),
        classifier=_Clf(),
    )
    det.min_confidence = 0.35
    assert det.detect(_png()) == []


def test_scene_fallback_when_yolo_blank(monkeypatch):
    class _Clf:
        def describe_person(self, _pil, _box):  # pragma: no cover
            raise AssertionError

        def classify_scene(self, _pil):
            return Detection("package", 0.81)

    det = _detector(
        monkeypatch,
        model=_fake_model([], [], []),
        classifier=_Clf(),
    )
    out = det.detect(_png())
    assert summarize(out) == "a package"


def test_no_classifier_gives_plain_person(monkeypatch):
    det = Detector(enable_clip=False)
    monkeypatch.setattr(det, "_load", lambda: _fake_model([0], [0.9], [[0, 0, 5, 5]]))
    out = det.detect(_png())
    assert [x.label for x in out] == ["person"]
    assert out[0].detail is None
    assert summarize(out) == "a person"
