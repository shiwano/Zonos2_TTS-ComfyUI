"""Emotion direction control for ZONOS2 speaker conditioning.

Ported from Zyphra/ZONOS2 (Apache-2.0), python/zonos2/tts/emotion.py, reduced
to the space="proj" path that the released emotion_directions/ ship with.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

logger = logging.getLogger("zonos2-tts-comfyui-emotion")

DIRECTIONS_DIR = Path(__file__).resolve().parent / "emotion_directions"
MANIFEST_NAME = "manifest.json"
CALIBRATION_NAME = "calibration.json"
AXIS_NAMES = ("valence", "arousal")
EMOTION_NONE = "none"
FALLBACK_EMOTION_NAMES = ("happy", "sad", "angry", "surprised")


@dataclass
class EmotionDirections:
    """Direction vectors for one speaker-conditioning space.

    Only ``space == "proj"`` is supported: the delta is added to the speaker
    vector after the model's speaker projection, so ``dim`` is the model's
    hidden size rather than the speaker-embedding size.
    """

    dim: int
    space: str
    named: dict[str, torch.Tensor] = field(default_factory=dict)
    axes: dict[str, torch.Tensor] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not self.named and not self.axes

    def vector(self, name: str) -> torch.Tensor | None:
        vector = self.named.get(name)
        if vector is None:
            vector = self.axes.get(name)
        return vector

    @classmethod
    def load(cls, directory: Path = DIRECTIONS_DIR) -> "EmotionDirections | None":
        manifest_path = Path(directory) / MANIFEST_NAME
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text())
        dim = int(manifest["dim"])
        space = str(manifest.get("space", "raw"))
        named: dict[str, torch.Tensor] = {}
        axes: dict[str, torch.Tensor] = {}
        for name, entry in manifest.get("directions", {}).items():
            vector = _load_vector(Path(directory) / entry["file"], dim, name)
            if str(entry.get("kind")) == "axis" or name in AXIS_NAMES:
                axes[name] = vector
            else:
                named[name] = vector
        return cls(dim=dim, space=space, named=named, axes=axes)


@dataclass
class EmotionCalibration:
    """Per-emotion strength chosen offline by the ZONOS2 authors.

    The shipped values are 3.0 for every direction except surprised, which is
    4.0. Folding them into the weights lets a single user-facing multiplier of
    1.0 mean "the calibrated amount" for every direction.
    """

    default: dict[str, float] = field(default_factory=dict)
    global_default: float = 1.0

    def strength(self, name: str) -> float:
        value = self.default.get(name)
        if value is None:
            return float(self.global_default)
        return float(value)

    @classmethod
    def load(cls, directory: Path = DIRECTIONS_DIR) -> "EmotionCalibration | None":
        path = Path(directory) / CALIBRATION_NAME
        if not path.is_file():
            return None
        data = json.loads(path.read_text())
        return cls(
            default={
                name: float(value)
                for name, value in data.get("default", {}).items()
            },
            global_default=float(data.get("global_default", 1.0)),
        )


def _load_vector(path: Path, expected_dim: int, name: str) -> torch.Tensor:
    array = np.asarray(np.load(path), dtype=np.float32).reshape(-1)
    if array.shape[0] != expected_dim:
        raise ValueError(
            f"Emotion direction '{name}' has dim {array.shape[0]}, expected "
            f"{expected_dim} ({path})."
        )
    return torch.from_numpy(np.ascontiguousarray(array))


_directions_cache: tuple[EmotionDirections | None] | None = None
_calibration_cache: tuple[EmotionCalibration | None] | None = None


def load_directions(
    directory: Path = DIRECTIONS_DIR,
) -> EmotionDirections | None:
    global _directions_cache
    if directory != DIRECTIONS_DIR:
        return EmotionDirections.load(directory)
    if _directions_cache is None:
        _directions_cache = (EmotionDirections.load(directory),)
    return _directions_cache[0]


def load_calibration(
    directory: Path = DIRECTIONS_DIR,
) -> EmotionCalibration | None:
    global _calibration_cache
    if directory != DIRECTIONS_DIR:
        return EmotionCalibration.load(directory)
    if _calibration_cache is None:
        _calibration_cache = (EmotionCalibration.load(directory),)
    return _calibration_cache[0]


def emotion_choices() -> list[str]:
    directions = load_directions()
    if directions is None or not directions.named:
        return [EMOTION_NONE, *FALLBACK_EMOTION_NAMES]
    return [EMOTION_NONE, *directions.named]


def emotion_hidden_delta(
    sliders: Mapping[str, float] | None = None,
    valence: float = 0.0,
    arousal: float = 0.0,
    strength: float = 1.0,
    directions: EmotionDirections | None = None,
    calibration: EmotionCalibration | None = None,
) -> torch.Tensor | None:
    """Combined hidden-space delta, or None when nothing is requested.

    The result is ``strength * sum(weight * calibrated * direction)`` as a 1-D
    float32 CPU tensor, to be added to the projected speaker vector inside the
    model.
    """
    directions = load_directions() if directions is None else directions
    if directions is None or directions.is_empty():
        return None
    if directions.space != "proj":
        raise ValueError(
            f"Emotion directions use space '{directions.space}'; this node "
            "only supports post-projection ('proj') directions."
        )
    calibration = load_calibration() if calibration is None else calibration

    weights: list[tuple[str, float]] = list((sliders or {}).items())
    weights.extend(zip(AXIS_NAMES, (valence, arousal)))

    delta = torch.zeros(directions.dim, dtype=torch.float32)
    requested = False
    for name, weight in weights:
        weight = float(weight)
        if weight == 0.0:
            continue
        vector = directions.vector(name)
        if vector is None:
            logger.warning(
                "Unknown ZONOS2 emotion direction %r; ignoring it.", name
            )
            continue
        if calibration is not None:
            weight *= calibration.strength(name)
        delta = delta + weight * vector
        requested = True

    if not requested or float(strength) == 0.0:
        return None
    return (float(strength) * delta).contiguous()
