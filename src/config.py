from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    return cfg


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--resume_mode", default=None, choices=["full", "weights"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--cache_rate", type=float, default=None)
    parser.add_argument("--val_cache_rate", type=float, default=None)
    parser.add_argument("--no_pin_memory", action="store_true",
                        help="Load batches without the pin-memory thread (data.pin_memory: false).")
    parser.add_argument("--force_missing_tabular", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow running on CPU (configs require CUDA).")
    parser.add_argument("--ignore_mismatched_shapes", action="store_true",
                        help="Skip checkpoint tensors whose shapes differ from the model (PanTS -> PANORAMA).")
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.max_epochs is not None:
        cfg.setdefault("train", {})["max_epochs"] = args.max_epochs
    if args.val_every is not None:
        cfg.setdefault("train", {})["val_every"] = args.val_every
    if args.save_every is not None:
        cfg.setdefault("train", {})["save_every"] = args.save_every
    if args.amp:
        cfg.setdefault("train", {})["amp"] = True
    if args.no_amp:
        cfg.setdefault("train", {})["amp"] = False
    if args.num_workers is not None:
        cfg.setdefault("data", {})["num_workers"] = args.num_workers
    if args.cache_rate is not None:
        cfg.setdefault("data", {})["cache_rate"] = args.cache_rate
    if getattr(args, "val_cache_rate", None) is not None:
        cfg.setdefault("data", {})["val_cache_rate"] = args.val_cache_rate
    if getattr(args, "no_pin_memory", False):
        cfg.setdefault("data", {})["pin_memory"] = False
    if args.force_missing_tabular:
        cfg.setdefault("tabular", {})["force_missing"] = True
    if getattr(args, "allow_cpu", False):
        cfg.setdefault("train", {})["require_cuda"] = False
        cfg.setdefault("infer", {})["require_cuda"] = False
    cfg["config_path"] = str(args.config)
    cfg["resume"] = bool(args.resume)
    if args.checkpoint_path is not None:
        cfg["checkpoint_path"] = args.checkpoint_path
    else:
        cfg.setdefault("checkpoint_path", None)
    if getattr(args, "ignore_mismatched_shapes", False):
        cfg["ignore_mismatched_shapes"] = True
    if args.resume_mode is not None:
        cfg["resume_mode"] = args.resume_mode
    else:
        # a checkpoint loaded without --resume only initialises the weights
        cfg["resume_mode"] = "full" if args.resume else "weights"
    return cfg
