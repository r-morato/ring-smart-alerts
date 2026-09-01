"""Local object detection on a snapshot image.

Two stages, both on CPU:

1. **YOLOv8n** (Ultralytics, COCO, 80 classes) finds the boxes -- people,
   vehicles, animals, common objects. The weights (~6 MB) download automatically
   on first use.
2. An optional **CLIP zero-shot** pass refines what YOLO cannot say on its own:

   * every ``person`` box is cropped and classified as ``adult`` / ``child`` /
     ``courier``, with a hedged ``(looks like a man/woman)`` guess;
   * when YOLO finds nothing at all, the whole frame is classified as
     ``package`` / ``animal`` (this catches foxes, raccoons and other critters
     COCO has no class for) / ``person`` / ``vehicle``.

The CLIP stage needs ``open-clip-torch``. If it is not installed, or
``ENABLE_CLIP`` is false, the detector silently falls back to plain YOLO.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolov8n.pt"

#: Cap on how many person crops we hand to CLIP per frame (latency bound).
MAX_PERSONS = 4

_UNSET = object()


@dataclass(frozen=True, slots=True)
class Detection:
    """A single recognised object.

    *label* is the coarse class used for logic (``person``, ``dog``, ``car``,
    ``package``, ``animal`` ...). *detail*, when present, is the refined human
    phrase shown to the user instead of the bare label -- e.g. ``"adult (looks
    like a man)"`` or ``"child"``.
    """

    label: str
    confidence: float
    detail: str | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.detail or self.label} ({self.confidence:.0%})"


class Detector:
    """Runs YOLOv8n on JPEG bytes and returns thresholded, refined labels."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        min_confidence: float = 0.35,
        *,
        enable_clip: bool = True,
        clip_model: str = "ViT-B-32-quickgelu",
        clip_pretrained: str = "openai",
    ) -> None:
        self.model_path = model_path
        self.min_confidence = min_confidence
        self.enable_clip = enable_clip
        self.clip_model = clip_model
        self.clip_pretrained = clip_pretrained
        self._model = None  # lazy: importing ultralytics/torch is slow
        self._clip: object | None = _UNSET  # lazy + "tried and unavailable" sentinel

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO  # deferred import

            logger.info("Loading YOLO model %s", self.model_path)
            self._model = YOLO(self.model_path)
        return self._model

    def _classifier(self) -> ClipClassifier | None:
        """Return the CLIP classifier, or ``None`` if unavailable/disabled."""
        if not self.enable_clip:
            return None
        if self._clip is _UNSET:
            try:
                import open_clip  # noqa: F401 - probe only
            except ImportError:
                logger.warning(
                    "ENABLE_CLIP is on but open-clip-torch is not installed; "
                    "person/scene refinement disabled"
                )
                self._clip = None
            else:
                self._clip = ClipClassifier(self.clip_model, self.clip_pretrained)
        return self._clip  # type: ignore[return-value]

    def detect(self, image: bytes) -> list[Detection]:
        """Detect objects in *image* (JPEG/PNG bytes).

        Returns detections above ``min_confidence`` sorted by confidence
        descending: one entry per ``person`` box (each individually refined) and
        one per other class.
        """
        from PIL import Image  # ultralytics pulls in Pillow

        model = self._load()
        pil = Image.open(io.BytesIO(image)).convert("RGB")
        results = model.predict(pil, verbose=False)

        persons: list[tuple[float, tuple[float, float, float, float]]] = []
        best: dict[str, float] = {}
        for result in results:
            names = result.names
            boxes = result.boxes
            for cls_id, conf, xyxy in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
            ):
                if conf < self.min_confidence:
                    continue
                label = names[int(cls_id)]
                if label == "person":
                    persons.append((conf, tuple(xyxy)))
                elif conf > best.get(label, 0.0):
                    best[label] = conf

        clf = self._classifier()

        detections: list[Detection] = []
        persons.sort(key=lambda p: p[0], reverse=True)
        for conf, box in persons[:MAX_PERSONS]:
            detail = clf.describe_person(pil, box) if clf is not None else None
            detections.append(Detection("person", conf, detail))

        detections += [Detection(label, conf) for label, conf in best.items()]
        detections.sort(key=lambda d: d.confidence, reverse=True)

        if not detections and clf is not None:
            fallback = clf.classify_scene(pil)
            if fallback is not None:
                detections = [fallback]
        return detections


class ClipClassifier:
    """Zero-shot image classification via open-clip, with cached text features.

    Prompt groups are constant, so their text embeddings are computed once and
    reused -- each call is then just one image forward pass plus a matmul.
    """

    #: "what kind of visitor" -- one softmax so the options compete honestly.
    #: The empty key is the escape hatch: when it wins we say nothing.
    _KIND = {
        "courier": "a delivery courier or postal worker in uniform, often with a parcel",
        "child": "a young child or a little kid, small and short",
        "adult": "a grown adult man or woman",
        "": "a person",
    }
    #: gender -- noisy, always surfaced hedged as "looks like ..."
    _GENDER = {
        "a man": "a photo of a man",
        "a woman": "a photo of a woman",
    }
    #: whole-frame fallback when YOLO found nothing
    _SCENE = {
        "package": "a cardboard delivery box or parcel left on a doorstep",
        "animal": "a wild animal such as a fox, raccoon, deer or a stray cat",
        "person": "a person standing near the door",
        "vehicle": "a car or a delivery truck",
        "nothing": "an empty porch or doorway, nothing notable",
    }

    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai") -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._text_cache: dict[tuple[str, ...], object] = {}

    def _load(self):
        if self._model is None:
            import open_clip

            logger.info("Loading CLIP model %s/%s", self.model_name, self.pretrained)
            model, _, preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            model.eval()
            self._model = model
            self._preprocess = preprocess
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
        return self._model

    def _text_features(self, texts: tuple[str, ...]):
        import torch

        cached = self._text_cache.get(texts)
        if cached is None:
            model = self._load()
            with torch.no_grad():
                feats = model.encode_text(self._tokenizer(list(texts)))
                feats = feats / feats.norm(dim=-1, keepdim=True)
            self._text_cache[texts] = feats
            cached = feats
        return cached

    def _zero_shot(self, pil, prompts: dict[str, str]) -> tuple[str, float]:
        """Return the best-matching key in *prompts* and its softmax probability."""
        import torch

        model = self._load()
        keys = tuple(prompts)
        text_feats = self._text_features(tuple(prompts[k] for k in keys))
        image_input = self._preprocess(pil).unsqueeze(0)
        with torch.no_grad():
            img_feats = model.encode_image(image_input)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)[0]
        idx = int(probs.argmax())
        return keys[idx], float(probs[idx])

    def describe_person(self, pil, box: tuple[float, float, float, float]) -> str | None:
        """Refine one ``person`` box into a phrase, or ``None`` if nothing sticks."""
        crop = _crop(pil, box)
        try:
            kind, kind_p = self._zero_shot(crop, self._KIND)
            gender, gender_p = self._zero_shot(crop, self._GENDER)
        except Exception as exc:  # noqa: BLE001 - torch/model errors are non-fatal
            logger.debug("CLIP person classify failed: %s", exc)
            return None

        # Conservative floors: a wrong "child" or "courier" is worse than a bare
        # "person". Tune these down once you've watched real events.
        floor = {"child": 0.65, "courier": 0.62}.get(kind, 0.45)
        base: str | None = kind if kind and kind_p >= floor else None

        hedge = f"looks like {gender}" if gender_p >= 0.60 else None

        if base and hedge and base != "courier":
            return f"{base} ({hedge})"
        if base:
            return base
        if hedge:
            return f"person ({hedge})"
        return None

    def classify_scene(self, pil) -> Detection | None:
        """Whole-frame guess for when YOLO returned nothing."""
        try:
            label, prob = self._zero_shot(pil, self._SCENE)
        except Exception as exc:  # noqa: BLE001
            logger.debug("CLIP scene classify failed: %s", exc)
            return None
        if label == "nothing" or prob < 0.45:
            return None
        return Detection(label, prob)


def _crop(pil, box: tuple[float, float, float, float], pad: float = 0.08):
    """Crop *box* out of *pil* with a small margin, clamped to the image."""
    width, height = pil.size
    left, top, right, bottom = box
    px, py = (right - left) * pad, (bottom - top) * pad
    return pil.crop(
        (
            max(0, int(left - px)),
            max(0, int(top - py)),
            min(width, int(right + px)),
            min(height, int(bottom + py)),
        )
    )


def summarize(detections: list[Detection]) -> str:
    """Turn detections into a short human phrase.

    ``[]`` -> "nothing recognised"
    ``[person]`` -> "a person"
    ``[person(detail="child"), dog]`` -> "a child and a dog"
    ``[person(detail="adult (looks like a man)"), person(detail="child")]``
        -> "an adult (looks like a man) and a child"
    """
    if not detections:
        return "nothing recognised"

    articled = [f"{_article(name)} {name}" for name in (d.detail or d.label for d in detections)]
    if len(articled) == 1:
        return articled[0]
    return ", ".join(articled[:-1]) + " and " + articled[-1]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"
