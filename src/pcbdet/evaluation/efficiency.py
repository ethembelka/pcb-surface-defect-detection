"""Model size and speed: parameters, GFLOPs and FPS (protocol: batch 1, FP16, warm-up + timed loop).

FPS depends on the hardware, so the GPU name is recorded next to the number. The measurement
covers the model forward pass incl. Ultralytics pre/post-processing (what an inspection system
would pay per image), using the same ``imgsz`` as training.
"""

from __future__ import annotations

import time
from pathlib import Path


def efficiency(weights: Path, sample_images: list[Path], imgsz: int, half: bool, warmup: int, iters: int) -> dict:
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    model = YOLO(str(weights))
    params = int(get_num_params(model.model))
    gflops = round(float(get_flops(model.model, imgsz)), 2)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    half = half and device != "cpu"

    imgs = [str(p) for p in sample_images]
    if not imgs:
        raise ValueError("efficiency(): no sample images")

    def run(n: int) -> float:
        if device != "cpu":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(n):
            model.predict(imgs[i % len(imgs)], imgsz=imgsz, half=half, device=device, verbose=False)
        if device != "cpu":
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    run(warmup)
    total = run(iters)
    ms = 1000 * total / iters
    return {"params": params, "gflops": gflops, "ms_per_image": round(ms, 2), "fps": round(1000 / ms, 1),
            "imgsz": imgsz, "half": half, "batch": 1, "device": gpu, "iters": iters}
