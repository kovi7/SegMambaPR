"""Training curves and per-case evaluation plots."""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class PlotConfig:
    loss_png: str = "loss_and_dice.png"
    dice_per_class_png: str = "dice_per_class.png"
    pdac_metrics_png: str = "pdac_metrics.png"
    lr_png: str = "learning_rate.png"
    # validation without tabular data is drawn on separate figures
    dice_per_class_missing_png: str = "dice_per_class_missing_tabular.png"
    loss_missing_png: str = "loss_and_dice_missing_tabular.png"
    pdac_metrics_missing_png: str = "pdac_metrics_missing_tabular.png"


def label_names_from_cfg(labels: Optional[Dict[str, int]]) -> Dict[int, str]:
    """Invert a config `labels: {name: index}` mapping to `{index: name}` for plot ticks."""
    if not labels:
        return {}
    return {int(v): str(k) for k, v in labels.items()}


def _load_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # a resumed run can log an epoch twice; keep the last occurrence
    if "epoch" in df.columns:
        df = df.drop_duplicates(subset="epoch", keep="last")
    df = df.sort_values("epoch").reset_index(drop=True)
    for col in df.columns:
        if col != "epoch":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _collect_finite(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    parts = []
    for c in cols:
        if c in df.columns:
            v = df[c].to_numpy(dtype=float, copy=False)
            v = v[np.isfinite(v)]
            if v.size:
                parts.append(v)
    return np.concatenate(parts) if parts else np.array([], dtype=float)


def _axis_limits(values: np.ndarray, bottom: float = 0.0, headroom: float = 0.05) -> tuple:
    if values.size == 0:
        return (bottom, bottom + 1.0)
    data_min = float(np.nanmin(values))
    data_max = float(np.nanmax(values))
    lo = min(bottom, data_min)
    span = max(data_max - lo, 1e-8)
    return (lo, data_max + headroom * span)


def _marker_style(n_points: int) -> dict:
    if n_points <= 30:
        return {"marker": "o", "markersize": 7, "linewidth": 2.0}
    if n_points <= 100:
        return {"marker": ".", "markersize": 5, "linewidth": 1.6}
    return {"marker": "", "linewidth": 1.5}


def _line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str,
    ylabel: str,
    save_path: str,
    secondary_y_cols: Optional[List[str]] = None,
    secondary_ylabel: Optional[str] = None,
    y_bottom: float = 0.0,
    secondary_y_bottom: float = 0.0,
    log_y: bool = False,
    labels: Optional[Dict[str, str]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = df[x_col]
    style = _marker_style(len(x))

    for col in y_cols:
        if col in df.columns and df[col].notna().any():
            y = df[col].to_numpy(dtype=float)
            mask = np.isfinite(y)
            ax.plot(x[mask], y[mask], linestyle="-", label=(labels or {}).get(col, col), **style)

    ax.set(xlabel=x_col, ylabel=ylabel, title=title)
    ax.grid(True, alpha=0.3)
    if len(x):
        x_min, x_max = float(x.min()), float(x.max())
        pad = max((x_max - x_min) * 0.01, 0.5)
        ax.set_xlim(x_min - pad, x_max + pad)

    if log_y:
        positive = _collect_finite(df, y_cols)
        positive = positive[positive > 0]
        if positive.size:
            ax.set_yscale("log")
            ax.set_ylim(positive.min() * 0.9, positive.max() * 1.5)
    else:
        ax.set_ylim(*_axis_limits(_collect_finite(df, y_cols), bottom=y_bottom))

    if secondary_y_cols:
        ax2 = ax.twinx()
        for col in secondary_y_cols:
            if col in df.columns and df[col].notna().any():
                y = df[col].to_numpy(dtype=float)
                mask = np.isfinite(y)
                ax2.plot(x[mask], y[mask], linestyle="--", label=(labels or {}).get(col, col), **style)
        ax2.set_ylabel(secondary_ylabel or "")
        ax2.set_ylim(*_axis_limits(_collect_finite(df, secondary_y_cols), bottom=secondary_y_bottom))
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc="best", fontsize=9)
    elif ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def _all_val_prefixes(df: pd.DataFrame) -> List[str]:
    suffix = "_dice_mean_fg"
    return sorted({c[: -len(suffix)] for c in df.columns if c.endswith(suffix)})


def _val_prefixes(df: pd.DataFrame) -> List[str]:
    """Validation prefixes for the main plots: clinical data available."""
    prefixes = [p for p in _all_val_prefixes(df) if "missing" not in p]
    return prefixes or ["val"]


def _missing_prefixes(df: pd.DataFrame) -> List[str]:
    """Validation prefixes for the forced-missing-tabular regime."""
    return [p for p in _all_val_prefixes(df) if "missing" in p]


def _class_display_labels(df_cols: List[str], prefixes: List[str], stem_suffix: str, label_names: Optional[Dict[int, str]]) -> Dict[str, str]:
    """Map columns like "val_dice_c1" -> a display label using label_names (e.g. "pdac"),
    falling back to "c1" if the class index isn't in label_names. Disambiguates by prefix
    only when more than one prefix is present (e.g. "val" vs a forced-missing variant)."""
    out: Dict[str, str] = {}
    for p in prefixes:
        stem = f"{p}_{stem_suffix}"
        for col in df_cols:
            if not col.startswith(stem):
                continue
            try:
                idx = int(col[len(stem):])
            except ValueError:
                continue
            name = (label_names or {}).get(idx, f"c{idx}")
            out[col] = name if len(prefixes) == 1 else f"{p} {name}"
    return out


def plot_training_curves(
    metrics_csv: str,
    output_dir: Optional[str] = None,
    cfg: Optional[PlotConfig] = None,
    label_names: Optional[Dict[int, str]] = None,
) -> Dict[str, str]:
    cfg = cfg or PlotConfig()
    df = _load_df(metrics_csv)
    output_dir = output_dir or os.path.dirname(metrics_csv) or "."
    os.makedirs(output_dir, exist_ok=True)

    out = {k: os.path.join(output_dir, v) for k, v in {
        "loss": cfg.loss_png,
        "dice_per_class": cfg.dice_per_class_png,
        "pdac_metrics": cfg.pdac_metrics_png,
        "learning_rate": cfg.lr_png,
    }.items()}

    pdac_metric_names = ["sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "auroc", "ap"]

    def draw(prefixes: List[str], keys: Dict[str, str], suffix: str) -> None:
        """Draw the per-class Dice, loss/mean-Dice and detection figures for one
        validation regime. ``keys`` maps figure name -> output path."""
        if not prefixes:
            return
        dice_cols = [c for p in prefixes for c in df.columns if c.startswith(f"{p}_dice_c")]
        if dice_cols:
            _line_plot(
                df, "epoch", dice_cols, f"Validation Dice per class{suffix}", "Dice",
                keys["dice_per_class"],
                labels=_class_display_labels(df.columns.tolist(), prefixes, "dice_c", label_names),
            )

        loss_cols = [c for c in ["train_loss"] if c in df.columns] + [f"{p}_loss" for p in prefixes if f"{p}_loss" in df.columns]
        dice_mean_cols = [f"{p}_dice_mean_fg" for p in prefixes if f"{p}_dice_mean_fg" in df.columns]
        if loss_cols or dice_mean_cols:
            _line_plot(
                df, "epoch", loss_cols,
                f"Training/validation loss and mean Dice{suffix}", "Loss", keys["loss"],
                secondary_y_cols=dice_mean_cols, secondary_ylabel="Dice",
            )

        pdac_cols = [f"{p}_pdac_{m}" for p in prefixes for m in pdac_metric_names if f"{p}_pdac_{m}" in df.columns]
        if pdac_cols:
            _line_plot(df, "epoch", pdac_cols, f"PDAC detection-from-segmentation metrics{suffix}", "Metric", keys["pdac_metrics"])

    draw(_val_prefixes(df), out, "")
    missing = _missing_prefixes(df)
    if missing:
        out.update({
            "dice_per_class_missing": os.path.join(output_dir, cfg.dice_per_class_missing_png),
            "loss_missing": os.path.join(output_dir, cfg.loss_missing_png),
            "pdac_metrics_missing": os.path.join(output_dir, cfg.pdac_metrics_missing_png),
        })
        draw(missing, {
            "dice_per_class": out["dice_per_class_missing"],
            "loss": out["loss_missing"],
            "pdac_metrics": out["pdac_metrics_missing"],
        }, " (tabular forced missing)")

    _line_plot(df, "epoch", ["lr"], "Learning rate", "LR", out["learning_rate"], log_y=True)

    return out


def _class_ticklabels(cols: List[str], stem: str, label_names: Optional[Dict[int, str]]) -> List[str]:
    """Map columns like "dice_c1" -> a display label using label_names (e.g. "pdac"),
    falling back to "c1" if the class index isn't in label_names."""
    out = []
    for c in cols:
        try:
            idx = int(c[len(stem):])
        except ValueError:
            out.append(c.replace(stem, ""))
            continue
        out.append((label_names or {}).get(idx, f"c{idx}"))
    return out


def plot_evaluation(evaluation_csv: str, output_dir: Optional[str] = None, label_names: Optional[Dict[int, str]] = None) -> Dict[str, str]:
    """Plot per-case inference evaluation results produced by src.evaluate."""
    if not os.path.exists(evaluation_csv):
        raise FileNotFoundError(f"Evaluation CSV not found: {evaluation_csv}")
    df = pd.read_csv(evaluation_csv)
    output_dir = output_dir or os.path.dirname(evaluation_csv) or "."
    os.makedirs(output_dir, exist_ok=True)

    out = {
        "dice_boxplot": os.path.join(output_dir, "eval_dice_per_class_boxplot.png"),
        "hd95_boxplot": os.path.join(output_dir, "eval_hd95_per_class_boxplot.png"),
        "pdac_confusion": os.path.join(output_dir, "eval_pdac_confusion_matrix.png"),
        "pdac_score_by_truth": os.path.join(output_dir, "eval_pdac_score_by_truth_boxplot.png"),
    }

    dice_cols = [c for c in df.columns if c.startswith("dice_c")]
    if dice_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [df[c].dropna().to_numpy(dtype=float) for c in dice_cols]
        ax.boxplot(data, tick_labels=_class_ticklabels(dice_cols, "dice_c", label_names), showmeans=True)
        ax.set(title="Per-case Dice by class", ylabel="Dice")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out["dice_boxplot"], dpi=160)
        plt.close(fig)

    hd_cols = [c for c in df.columns if c.startswith("hd95_c")]
    if hd_cols:
        fig, ax = plt.subplots(figsize=(8, 5))
        data = [df[c].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float) for c in hd_cols]
        ax.boxplot(data, tick_labels=_class_ticklabels(hd_cols, "hd95_c", label_names), showmeans=True)
        ax.set(title="Per-case HD95 by class", ylabel="HD95 (mm)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out["hd95_boxplot"], dpi=160)
        plt.close(fig)

    if {"pred_has_pdac", "true_has_pdac"}.issubset(df.columns):
        conf = np.zeros((2, 2), dtype=int)
        for t, p in zip(df["true_has_pdac"].astype(int), df["pred_has_pdac"].astype(int)):
            conf[t, p] += 1
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(conf, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(conf[i, j]), ha="center", va="center")
        ax.set(
            xticks=[0, 1], yticks=[0, 1],
            xticklabels=["no_pdac", "pdac"], yticklabels=["no_pdac", "pdac"],
            xlabel="Predicted", ylabel="True", title="PDAC detection confusion matrix",
        )
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(out["pdac_confusion"], dpi=160)
        plt.close(fig)

    if {"pdac_score", "true_has_pdac"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6, 5))
        groups = [df.loc[df["true_has_pdac"] == t, "pdac_score"].dropna().to_numpy(dtype=float) for t in (0, 1)]
        ax.boxplot(groups, tick_labels=["no_pdac", "pdac"], showmeans=True)
        ax.set(title="Predicted PDAC score by ground truth", ylabel="pdac_score (fraction of voxels)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out["pdac_score_by_truth"], dpi=160)
        plt.close(fig)

    return out


def _cli() -> None:
    p = argparse.ArgumentParser(description="Plot training curves and/or inference evaluation results.")
    p.add_argument("--metrics_csv", default=None, help="Path to metrics.csv produced by src.train.")
    p.add_argument("--evaluation_csv", default=None, help="Path to evaluation.csv produced by src.evaluate.")
    p.add_argument("--output_dir", default=None, help="Output directory for plots. Defaults next to the input CSV.")
    p.add_argument("--config", default=None, help="Optional training config; its `labels` mapping is used to label per-class plot ticks with real names (e.g. pdac, veins) instead of c1, c2.")
    args = p.parse_args()
    if not args.metrics_csv and not args.evaluation_csv:
        raise SystemExit("Provide --metrics_csv and/or --evaluation_csv.")
    label_names: Dict[int, str] = {}
    if args.config:
        from .config import load_config
        label_names = label_names_from_cfg(load_config(args.config).get("labels"))
    if args.metrics_csv:
        paths = plot_training_curves(args.metrics_csv, output_dir=args.output_dir, label_names=label_names)
        print("Saved training plots:")
        for name, path in paths.items():
            print(f"  {name}: {path}")
    if args.evaluation_csv:
        paths = plot_evaluation(args.evaluation_csv, output_dir=args.output_dir, label_names=label_names)
        print("Saved evaluation plots:")
        for name, path in paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    _cli()
