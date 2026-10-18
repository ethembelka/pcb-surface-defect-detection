"""Model-name -> ``Detector`` class lookup.

The trainer asks the registry for a detector by the ``model`` string of the train config
(``yolo11s.pt`` -> Ultralytics). Adding a new family = one class + one line here.
"""

from __future__ import annotations

from pcbdet.models.base import Detector
from pcbdet.models.ultralytics_detector import UltralyticsDetector

# prefix of the model name -> detector class
_REGISTRY: dict[str, type] = {
    "yolo": UltralyticsDetector,
    "rtdetr": UltralyticsDetector,  # RT-DETR is also served by Ultralytics (phase A5, optional)
}


def get_detector(model_name: str) -> Detector:
    key = model_name.lower().split("/")[-1].split("\\")[-1]
    for prefix, cls in _REGISTRY.items():
        if key.startswith(prefix):
            return cls()
    raise KeyError(f"no detector registered for model '{model_name}' (known prefixes: {list(_REGISTRY)})")
