from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def report_conv_backend(model, roi, dtype, use_amp: bool, channels_last: bool) -> str:
    """Report whether a 3D convolution runs on cuDNN or on the slow im2col fallback.

    Under half-precision autocast, contiguous 3D tensors have no cuDNN kernel and fall
    back to im2col. One convolution at the first encoder stage's shape is profiled once.
    """
    if not torch.cuda.is_available():
        return "skipped (no GPU)"
    try:
        from torch.profiler import ProfilerActivity, profile

        channels = 48
        size = max(int(roi[0]) // 2, 8)
        conv = torch.nn.Conv3d(channels, channels, 3, padding=1, bias=False).cuda()
        x = torch.randn(1, channels, size, size, size, device="cuda")
        if channels_last:
            conv = conv.to(memory_format=torch.channels_last_3d)
            x = x.contiguous(memory_format=torch.channels_last_3d)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            with torch.autocast("cuda", dtype=dtype, enabled=use_amp):
                conv(x).sum().backward()
            torch.cuda.synchronize()
        names = {e.key for e in prof.key_averages()}
        del conv, x
        torch.cuda.empty_cache()
        if any("cudnn" in n for n in names):
            return "cuDNN"
        if any("slow_conv" in n or "vol2col" in n for n in names):
            return "im2col fallback, several times slower; set train.channels_last"
        return "unknown backend"
    except Exception as exc:
        return f"check failed ({type(exc).__name__})"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        ensure_dir(self.path.parent)
        self.fieldnames = None
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                self.fieldnames = next(reader, None)

    def truncate_from_epoch(self, start_epoch: int) -> int:
        """Drop logged rows at or after ``start_epoch`` (used on resume); return their count."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return 0
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        kept = [r for r in rows if r.get("epoch") not in (None, "") and int(float(r["epoch"])) < int(start_epoch)]
        dropped = len(rows) - len(kept)
        if dropped <= 0:
            return 0
        with open(self.path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames or list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(kept)
        return dropped

    def log(self, row: Dict[str, Any]) -> None:
        flat = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                v = v.item()
            elif isinstance(v, torch.Tensor):
                v = v.detach().cpu().item() if v.numel() == 1 else str(tuple(v.shape))
            flat[k] = v
        rewrite = False
        if self.fieldnames is None:
            self.fieldnames = list(flat.keys())
            write_header = True
        else:
            for k in flat.keys():
                if k not in self.fieldnames:
                    self.fieldnames.append(k)
                    rewrite = True
            write_header = not self.path.exists() or self.path.stat().st_size == 0
        if rewrite and self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "r", newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f))
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(old_rows)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(flat)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    scaler: Optional[Any],
    epoch: int,
    global_step: int,
    best_metric: float,
    config: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None and hasattr(scheduler, "state_dict") else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
    }
    if extras:
        payload.update(extras)
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
    resume_mode: str = "full",
    ignore_mismatched_shapes: bool = False,
) -> Dict[str, Any]:
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("model", ckpt)
    if ignore_mismatched_shapes:
        current = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in state.items():
            if k in current and tuple(current[k].shape) == tuple(v.shape):
                filtered[k] = v
            else:
                skipped.append(k)
        state = filtered
        print(f"Ignoring {len(skipped)} checkpoint tensors with mismatched or missing shapes from {path}.")
        if skipped:
            print("First skipped keys:", skipped[:20])
    missing, unexpected = model.load_state_dict(state, strict=strict)
    print(f"Loaded model weights from {path}; missing={len(missing)}, unexpected={len(unexpected)}")
    if resume_mode == "full":
        if ignore_mismatched_shapes:
            print("ignore_mismatched_shapes=True: not loading optimizer/scheduler/scaler even if resume_mode=full.")
        else:
            if optimizer is not None and ckpt.get("optimizer") is not None:
                optimizer.load_state_dict(ckpt["optimizer"])
            if scheduler is not None and ckpt.get("scheduler") is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(ckpt["scheduler"])
            if scaler is not None and ckpt.get("scaler") is not None:
                scaler.load_state_dict(ckpt["scaler"])
    return ckpt


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}
