"""Local object detection on a snapshot image.

Uses Ultralytics YOLOv8n (COCO, 80 classes) which runs comfortably on CPU on a
Raspberry Pi / home server. The model weights (~6 MB) download automatically on
first use and are cached by Ultralytics.

Package detection
-----------------
COCO has no "package"/"parcel" class, so a delivered box on the doorstep is not
recognised by v1. A zero-shot CLIP fallback is the intended extension point --
see :meth:`Detector._package_fallback`, which is a documented stub for now.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.pt"


@dataclass(frozen=True, slots=True)
class Detection:
    """A single recognised object."""

    label: str
    confidence: float

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.label} ({self.confidence:.0%})"


class Detector:
    """Runs YOLOv8n on JPEG bytes and returns thresholded, de-duplicated labels."""

    def __init__(self, model_path: str = DEFAULT_MODEL, min_confidence: float = 0.35) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._model = None  # lazy: importing ultralytics/torch is slow

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # deferred import

            logger.info("Loading YOLO model %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, image: bytes) -> list[Detection]:
        """Detect objects in *image* (JPEG/PNG bytes).

        Returns detections above ``min_confidence``, one per class, sorted by
        confidence descending.
        """
        from PIL import Image  # ultralytics pulls in Pillow

        model = self._load()
        pil = Image.open(io.BytesIO(image)).convert("RGB")
        results = model.predict(pil, verbose=False)

        best: dict[str, float] = {}
        for result in results:
            names = result.names
            for cls_id, conf in zip(
                result.boxes.cls.tolist(), result.boxes.conf.tolist(), strict=True
            ):
                label = names[int(cls_id)]
                if conf >= self.min_confidence and conf > best.get(label, 0.0):
                    best[label] = conf

        detections = [Detection(label, conf) for label, conf in best.items()]
        detections.sort(key=lambda d: d.confidence, reverse=True)

        if not detections:
            detections = self._package_fallback(image)
        return detections

    def _package_fallback(self, image: bytes) -> list[Detection]:  # noqa: ARG002
        """Stub for a CLIP zero-shot "is there a package on the doorstep?" check.

        Intended implementation: run open-clip / transformers CLIP with prompts
        like ["a cardboard delivery box on a doorstep", "an empty doorstep"] and
        emit ``Detection("package", score)`` when the box prompt wins. Not
        implemented in v1; returns nothing.
        """
        return []


def summarize(detections: list[Detection]) -> str:
    """Turn detections into a short human phrase.

    ``[]`` -> "nothing recognised"
    ``[person]`` -> "a person"
    ``[person, dog]`` -> "a person and a dog"
    ``[person, dog, car]`` -> "a person, a dog and a car"
    """
    if not detections:
        return "nothing recognised"

    articled = [f"{_article(d.label)} {d.label}" for d in detections]
    if len(articled) == 1:
        return articled[0]
    return ", ".join(articled[:-1]) + " and " + articled[-1]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"
