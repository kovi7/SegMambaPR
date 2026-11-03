from __future__ import annotations

import faulthandler
import os
import signal
import sys
import threading
import time
from typing import Dict, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .config import apply_cli_overrides, load_config, parse_train_args
from .data import build_loaders
from .losses import SegmentationLoss, build_vessel_band
from .models.factory import build_model
from .models.tab_gating import set_tabular_dropout
from .plot_metrics import label_names_from_cfg, plot_training_curves
from .utils import CSVLogger, count_parameters, ensure_dir, load_checkpoint, report_conv_backend, save_checkpoint, save_json, set_seed
from .validation import validate


def _make_grad_scaler(device_type: str, enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.GradScaler(device_type, enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast(device_type: str, dtype: torch.dtype, enabled: bool):
    if hasattr(torch, "amp"):
        return torch.amp.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled, dtype=dtype)


def _resolve_amp_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"fp32", "float32", "none"}:
        return torch.float32
    raise ValueError(f"Unsupported train.amp_dtype={name!r}")


_NO_DECAY_PARAM_NAMES = {"a_log", "d", "d_b", "d_s"}


def _build_optimizer(model: torch.nn.Module, cfg: Dict) -> AdamW:
    """AdamW without weight decay for biases, norms and the Mamba parameters A_log and D."""
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lname = name.lower()
        last = lname.rsplit(".", 1)[-1]
        no_wd = (
            param.ndim <= 1
            or name.endswith(".bias")
            or "norm" in lname
            or "all_missing" in lname
            or last in _NO_DECAY_PARAM_NAMES
        )
        (no_decay if no_wd else decay).append(param)
    wd = float(cfg["train"].get("weight_decay", 1e-5))
    lr = float(cfg["train"].get("lr", 2e-4))
    return AdamW(
        [
            {"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def _tabular_dropout_for_epoch(cfg: Dict, epoch: int, max_epochs: int) -> Optional[float]:
    """Tabular dropout ramped linearly from ``tabular.dropout`` to ``tabular.dropout_final``.

    Returns None when ``dropout_final`` is not set, which keeps the rate fixed.
    """
    tab_cfg = cfg.get("tabular", {})
    final = tab_cfg.get("dropout_final")
    if final is None:
        return None
    start = float(tab_cfg.get("dropout", 0.0))
    final = float(final)
    if max_epochs <= 1:
        return final
    frac = min(1.0, max(0.0, epoch / (max_epochs - 1)))
    return start + (final - start) * frac


def _build_scheduler(optimizer: AdamW, cfg: Dict, max_epochs: int):
    """Cosine annealing after a linear warmup of ``train.warmup_epochs`` (0 disables it)."""
    min_lr = float(cfg["train"].get("min_lr", 1e-6))
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 0))
    if warmup_epochs <= 0:
        return CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=min_lr)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, max_epochs - warmup_epochs), eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


def _validation_passes(cfg: Dict) -> list:
    """Validation passes to run: "normal" (with metadata), "missing" (metadata removed), or both."""
    val_cfg = cfg.get("val", {}) or {}
    passes = val_cfg.get("passes")
    if passes is None:
        legacy = bool(val_cfg.get("evaluate_forced_missing", False)
                      or val_cfg.get("run_forced_missing", False)
                      or val_cfg.get("evaluate_missing_tabular", False))
        passes = ["normal", "missing"] if legacy else ["normal"]
    passes = [str(x).lower() for x in passes]
    unknown = [x for x in passes if x not in {"normal", "missing"}]
    if unknown:
        raise ValueError(f"Unknown val.passes entries {unknown}; use 'normal' and/or 'missing'.")
    if not passes:
        raise ValueError("val.passes must name at least one pass.")
    return passes


class _StallWatchdog:
    """End the process when no batch has arrived for ``train.stall_minutes`` (0 disables it).

    Every batch re-arms a faulthandler timer. Shortly before it fires, the state of the main
    process and of every DataLoader worker is printed. The process then exits with stack
    traces, so a chained job can resume from last.ckpt.
    """

    def __init__(self, cfg: Dict, loaders=()):
        train_cfg = cfg.get("train", {}) or {}
        self.batch_s = 60.0 * float(train_cfg.get("stall_minutes", 45))
        self.first_s = 60.0 * float(train_cfg.get("stall_first_batch_minutes", 90))
        self.enabled = self.batch_s > 0
        self.loaders = tuple(loaders)
        self._last = time.monotonic()
        self._limit = float("inf")
        self._reported = False
        if self.enabled:
            threading.Thread(target=self._report_loop, name="stall-report", daemon=True).start()

    def arm(self, first: bool = False) -> None:
        if self.enabled:
            self._limit = self.first_s if first else self.batch_s
            self._last = time.monotonic()
            self._reported = False
            faulthandler.dump_traceback_later(self._limit, exit=True, file=sys.stderr)

    def disarm(self) -> None:
        if self.enabled:
            self._limit = float("inf")
            faulthandler.cancel_dump_traceback_later()

    def _report_loop(self) -> None:
        while True:
            time.sleep(min(30.0, max(self.batch_s / 10, 0.5)))
            lead = min(300.0, 0.2 * self._limit)
            idle = time.monotonic() - self._last
            if not self._reported and idle >= self._limit - lead:
                self._reported = True
                try:
                    self._report(idle)
                except Exception as exc:
                    print(f"[stall] report failed: {exc}", file=sys.stderr, flush=True)

    def _worker_pids(self):
        pids = []
        for loader in self.loaders:
            for w in getattr(getattr(loader, "_iterator", None), "_workers", None) or []:
                if getattr(w, "pid", None):
                    pids.append(w.pid)
        return pids

    @staticmethod
    def _proc_state(pid: int) -> str:
        try:
            with open(f"/proc/{pid}/stat") as fh:
                state = fh.read().rsplit(")", 1)[1].split()[0]
            with open(f"/proc/{pid}/wchan") as fh:
                return f"state {state}, wchan {fh.read().strip() or '-'}"
        except OSError:
            return "no /proc entry (process gone)"

    def _report(self, idle: float) -> None:
        print(f"[stall] no batch for {idle / 60:.1f} min, limit {self._limit / 60:.0f} min; "
              f"main process {self._proc_state(os.getpid())}", file=sys.stderr, flush=True)
        usr2 = getattr(signal, "SIGUSR2", None)
        for pid in self._worker_pids():
            print(f"[stall] worker {pid}: {self._proc_state(pid)}", file=sys.stderr, flush=True)
            if usr2 is not None:
                try:
                    os.kill(pid, usr2)
                except OSError:
                    pass
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)

    def wrap(self, loader):
        """Wrap the validation loader so that every case re-arms the timer."""
        return _ArmedLoader(loader, self)


class _ArmedLoader:
    def __init__(self, loader, watchdog: _StallWatchdog):
        self.loader, self.watchdog = loader, watchdog

    def __iter__(self):
        for batch in self.loader:
            self.watchdog.arm()
            yield batch

    def __len__(self) -> int:
        return len(self.loader)


def main() -> None:
    args = parse_train_args()
    cfg = apply_cli_overrides(load_config(args.config), args)
    set_seed(int(cfg.get("seed", 42)))
    output_dir = ensure_dir(cfg.get("output_dir", "output/run"))

    if bool(cfg.get("train", {}).get("require_cuda", True)) and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this config, but torch.cuda.is_available() is False.")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, tabproc = build_loaders(cfg)
    cfg.setdefault("tabular", {})["tab_dim"] = tabproc.dim
    cfg.setdefault("tabular", {})["preprocess_state"] = tabproc.state_dict()
    save_json(cfg, output_dir / "config_resolved.json")
    save_json(tabproc.state_dict(), output_dir / "tabular_preprocess.json")

    model = build_model(cfg, tab_dim=tabproc.dim).to(device)
    # channels_last_3d only pays off with half-precision autocast; the reported runs use fp32
    use_channels_last = bool(cfg["train"].get("channels_last", False)) and device.type == "cuda"
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    print(count_parameters(model))
    loss_fn = SegmentationLoss(
        num_classes=int(cfg.get("num_classes", cfg["model"].get("out_channels", 6))),
        vessel_band=build_vessel_band(cfg.get("loss", {}).get("vessel_band")),
    )
    optimizer = _build_optimizer(model, cfg)
    max_epochs = int(cfg["train"].get("max_epochs", 300))
    scheduler = _build_scheduler(optimizer, cfg, max_epochs)
    amp_dtype = _resolve_amp_dtype(cfg["train"].get("amp_dtype", "fp32"))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda" and amp_dtype != torch.float32
    scaler = _make_grad_scaler(device.type, enabled=use_amp and amp_dtype == torch.float16)
    print(f"3D convolutions: {report_conv_backend(model, cfg['data'].get('roi_size', [128, 128, 128]), amp_dtype, use_amp, use_channels_last)}", flush=True)
    logger = CSVLogger(output_dir / "metrics.csv")

    stop_requested: Dict[str, bool] = {"value": False}

    def _handle_usr1(signum, frame):
        print(f"Received signal {signum}. Training will save a checkpoint and stop safely.", flush=True)
        stop_requested["value"] = True

    try:
        signal.signal(signal.SIGUSR1, _handle_usr1)
    except Exception as exc:
        print(f"Could not register SIGUSR1 handler: {exc}", flush=True)
    # SIGUSR2 prints the stack of every thread and lets the run continue
    try:
        faulthandler.register(signal.SIGUSR2, file=sys.stderr, all_threads=True)
    except Exception as exc:
        print(f"Could not register SIGUSR2 stack dump: {exc}", flush=True)
    watchdog = _StallWatchdog(cfg, loaders=(train_loader, val_loader))

    start_epoch = 0
    global_step = 0
    best_metric = -1e9
    ckpt_path = cfg.get("checkpoint_path")
    if cfg.get("resume"):
        ckpt_path = ckpt_path or str(output_dir / "last.ckpt")
    if ckpt_path:
        ckpt = load_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            scaler,
            map_location=device,
            strict=False,
            resume_mode=cfg.get("resume_mode", "weights"),
            ignore_mismatched_shapes=bool(cfg.get("checkpoint", {}).get("ignore_mismatched_shapes", False) or cfg.get("train", {}).get("ignore_mismatched_shapes", False) or cfg.get("ignore_mismatched_shapes", False)),
        )
        if cfg.get("resume_mode", "weights") == "full":
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            global_step = int(ckpt.get("global_step", 0))
            best_metric = float(ckpt.get("best_metric", best_metric))
            dropped = logger.truncate_from_epoch(start_epoch)
            if dropped:
                print(f"Resuming at epoch {start_epoch}: dropped {dropped} metrics.csv row(s) from epochs that will be re-trained.", flush=True)

    val_every = int(cfg["train"].get("val_every", 10))
    save_every = int(cfg["train"].get("save_every", 25))
    plot_every = int(cfg["train"].get("plot_every", 10))
    grad_accum = int(cfg["train"].get("grad_accum", 1))
    best_key = cfg["train"].get("best_metric", "val_dice_mean_fg")
    if int(cfg["train"].get("val_every", 10)) <= 0:
        print("val_every <= 0: validation disabled, no best.ckpt will be written.", flush=True)
    val_passes = _validation_passes(cfg)
    needed = "missing" if best_key.startswith("val_missing_") else "normal"
    if needed not in val_passes:
        raise ValueError(
            f"train.best_metric={best_key!r} is not produced by val.passes={val_passes}. "
            "A forced-missing metric needs 'missing' in val.passes."
        )
    extras = {"tabular_preprocess": tabproc.state_dict()}

    for epoch in range(start_epoch, max_epochs):
        epoch_start = time.time()
        model.train()
        tab_p = _tabular_dropout_for_epoch(cfg, epoch, max_epochs)
        if tab_p is not None:
            set_tabular_dropout(model, tab_p)
        epoch_loss = torch.zeros((), device=device, dtype=torch.float32)
        steps = 0
        accum_steps = 0
        optimizer.zero_grad(set_to_none=True)
        watchdog.arm(first=(epoch == start_epoch))
        for batch in train_loader:
            image = batch["image"].to(device, non_blocking=True)
            if use_channels_last:
                image = image.contiguous(memory_format=torch.channels_last_3d)
            label = batch["label"].to(device, non_blocking=True)
            tab = batch["tabular"].float().to(device, non_blocking=True)
            tab_mask = batch["tabular_mask"].float().to(device, non_blocking=True)
            with _autocast(device.type, amp_dtype, enabled=use_amp):
                out = model(image, tabular=tab, tabular_mask=tab_mask)
                loss_raw = loss_fn(out["seg_logits"], label)
                loss = loss_raw / grad_accum
            scaler.scale(loss).backward()
            accum_steps += 1
            if accum_steps == grad_accum:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 12.0)))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                accum_steps = 0
            epoch_loss += loss_raw.detach().float()
            steps += 1
            watchdog.arm()
            if stop_requested["value"]:
                watchdog.disarm()
                if accum_steps > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 12.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    accum_steps = 0
                save_checkpoint(output_dir / "last.ckpt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg, extras=extras)
                save_checkpoint(output_dir / "interrupt.ckpt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg, extras=extras)
                logger.log({"epoch": epoch, "global_step": global_step, "train_loss": float(epoch_loss) / max(steps, 1), "lr": scheduler.get_last_lr()[0], "interrupted": 1})
                print(f"Saved interrupt checkpoints to {output_dir}.", flush=True)
                return

        if accum_steps > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg["train"].get("grad_clip", 12.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        scheduler.step()
        train_seconds = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "epoch_seconds": round(train_seconds, 1),
            "global_step": global_step,
            "train_loss": float(epoch_loss) / max(steps, 1),
            "lr": scheduler.get_last_lr()[0],
        }
        val_start = time.time()
        if val_every > 0 and (epoch % val_every == 0 or epoch == max_epochs - 1):
            for pass_name in val_passes:
                watchdog.arm(first=True)
                row.update(validate(
                    model, watchdog.wrap(val_loader), loss_fn, cfg, device,
                    prefix="val_missing" if pass_name == "missing" else "val",
                    force_missing_tabular=(pass_name == "missing"),
                ))
            watchdog.arm()
            metric = float(row[best_key])
            if metric > best_metric:
                best_metric = metric
                save_checkpoint(output_dir / "best.ckpt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg, extras=extras)
            row["val_seconds"] = round(time.time() - val_start, 1)
        if epoch % save_every == 0 or epoch == max_epochs - 1:
            save_checkpoint(output_dir / "last.ckpt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg, extras=extras)
        logger.log(row)
        print(row, flush=True)
        if plot_every > 0 and (epoch % plot_every == 0 or epoch == max_epochs - 1):
            try:
                plot_training_curves(str(output_dir / "metrics.csv"), output_dir=str(output_dir), label_names=label_names_from_cfg(cfg.get("labels")))
            except Exception as exc:
                print(f"Plotting failed (continuing training): {exc}", flush=True)
        if stop_requested["value"]:
            watchdog.disarm()
            save_checkpoint(output_dir / "last.ckpt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg, extras=extras)
            save_checkpoint(output_dir / "interrupt.ckpt", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, cfg, extras=extras)
            print(f"Saved interrupt checkpoints to {output_dir}.", flush=True)
            return
    watchdog.disarm()


if __name__ == "__main__":
    main()
