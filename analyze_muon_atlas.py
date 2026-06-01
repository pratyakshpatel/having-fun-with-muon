#!/usr/bin/env python3
"""Aggregate Muon Routing Atlas runs and produce report plots/tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


FIGURES = [
    "validation_loss_vs_tokens_tinystories.png",
    "validation_loss_vs_tokens_finewebedu.png",
    "validation_loss_vs_wallclock.png",
    "final_loss_barplot.png",
    "routing_heatmap.png",
    "layerwise_heatmap.png",
    "update_effective_rank_by_module.png",
    "update_norm_by_module.png",
    "singular_value_spectrum_examples.png",
    "qk_logit_growth.png",
    "batch_size_scaling.png",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--report_dir", default="report")
    return p.parse_args()


def load_runs(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    metric_frames, geom_frames, summaries = [], [], []
    for run in sorted(runs_dir.glob("*")):
        if not run.is_dir():
            continue
        cfg_path = run / "config_resolved.yaml"
        metrics_path = run / "metrics.csv"
        cfg = {}
        if cfg_path.exists():
            try:
                cfg = yaml.safe_load(cfg_path.read_text()) or {}
            except Exception as exc:
                print(f"WARNING: could not read {cfg_path}: {exc}")
        if metrics_path.exists():
            try:
                df = pd.read_csv(metrics_path)
                for key in ["dataset", "model_size", "routing", "seed", "run_dir"]:
                    df[key] = cfg.get(key, run.name)
                metric_frames.append(df)
                valid = df[df["val_loss"].notna()]
                if not valid.empty:
                    best = valid.loc[valid["val_loss"].idxmin()]
                    last = valid.iloc[-1]
                    summaries.append(
                        {
                            "run": run.name,
                            "dataset": cfg.get("dataset", ""),
                            "model_size": cfg.get("model_size", ""),
                            "routing": cfg.get("routing", ""),
                            "seed": cfg.get("seed", ""),
                            "best_val_loss": best["val_loss"],
                            "best_checkpoint_step": int(best["step"]),
                            "final_val_loss": last["val_loss"],
                            "tokens_seen": int(last["tokens_seen"]),
                            "wall_clock_seconds": float(last["wall_clock_seconds"]),
                            "tokens_per_sec": float(last["tokens_per_sec"]),
                            "peak_memory": float(df.get("gpu_memory_reserved_gb", pd.Series([0])).max()),
                        }
                    )
            except Exception as exc:
                print(f"WARNING: could not aggregate {metrics_path}: {exc}")
        geom_path = run / "geometry_metrics.csv"
        if geom_path.exists():
            try:
                gf = pd.read_csv(geom_path)
                for key in ["dataset", "model_size", "routing", "seed", "run_dir"]:
                    gf[key] = cfg.get(key, run.name)
                geom_frames.append(gf)
            except Exception as exc:
                print(f"WARNING: could not aggregate {geom_path}: {exc}")
    metrics = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    geometry = pd.concat(geom_frames, ignore_index=True) if geom_frames else pd.DataFrame()
    return metrics, geometry, summaries


def placeholder(path: Path, message: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.axis("off")
    plt.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def line_plot(metrics: pd.DataFrame, dataset: str, path: Path) -> None:
    sub = metrics[(metrics["dataset"] == dataset) & metrics["val_loss"].notna()]
    if sub.empty:
        placeholder(path, f"No completed runs were found for {dataset} at report generation time.")
        return
    plt.figure(figsize=(8, 5))
    for (routing, seed), g in sub.groupby(["routing", "seed"]):
        plt.plot(g["tokens_seen"], g["val_loss"], label=f"{routing} s{seed}", alpha=0.8)
    plt.xlabel("Tokens seen")
    plt.ylabel("Validation loss")
    plt.title(dataset)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def wallclock_plot(metrics: pd.DataFrame, path: Path) -> None:
    sub = metrics[metrics["val_loss"].notna()]
    if sub.empty:
        placeholder(path, "No completed runs were found for this figure at report generation time.")
        return
    plt.figure(figsize=(8, 5))
    for routing, g in sub.groupby("routing"):
        mean = g.groupby("wall_clock_seconds")["val_loss"].mean().reset_index()
        plt.plot(mean["wall_clock_seconds"] / 3600.0, mean["val_loss"], label=routing)
    plt.xlabel("Wall-clock hours")
    plt.ylabel("Validation loss")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def final_bar(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        placeholder(path, "No completed runs were found for this figure at report generation time.")
        return
    agg = summary.groupby(["dataset", "routing"])["final_val_loss"].agg(["mean", "std"]).reset_index()
    labels = agg["dataset"] + "\n" + agg["routing"]
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(len(agg)), agg["mean"], yerr=agg["std"].fillna(0.0))
    plt.xticks(np.arange(len(agg)), labels, rotation=45, ha="right", fontsize=8)
    plt.ylabel("Final validation loss")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def heatmap(summary: pd.DataFrame, path: Path, layerwise: bool = False) -> None:
    if summary.empty:
        placeholder(path, "No completed runs were found for this figure at report generation time.")
        return
    sub = summary.copy()
    if layerwise:
        sub = sub[sub["routing"].str.contains("early|middle|late", regex=True, na=False)]
    else:
        sub = sub[~sub["routing"].str.contains("early|middle|late", regex=True, na=False)]
    if sub.empty:
        placeholder(path, "This experiment was not run yet.")
        return
    piv = sub.pivot_table(index="routing", columns="dataset", values="final_val_loss", aggfunc="mean")
    plt.figure(figsize=(7, max(3, 0.4 * len(piv))))
    plt.imshow(piv.values, aspect="auto", cmap="viridis_r")
    plt.colorbar(label="Final validation loss")
    plt.yticks(np.arange(len(piv.index)), piv.index)
    plt.xticks(np.arange(len(piv.columns)), piv.columns)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def geometry_plot(geometry: pd.DataFrame, value: str, path: Path) -> None:
    if geometry.empty or value not in geometry:
        placeholder(path, "No geometry metrics were found for this figure at report generation time.")
        return
    agg = geometry.groupby(["routing", "module_name"])[value].mean().reset_index()
    labels = agg["routing"] + "\n" + agg["module_name"]
    plt.figure(figsize=(11, 5))
    plt.bar(np.arange(len(agg)), agg[value])
    plt.xticks(np.arange(len(agg)), labels, rotation=60, ha="right", fontsize=7)
    plt.ylabel(value.replace("_", " "))
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def simple_placeholder_set(fig_dir: Path) -> None:
    for name in ["singular_value_spectrum_examples.png", "qk_logit_growth.png", "batch_size_scaling.png"]:
        placeholder(fig_dir / name, "This experiment was not run yet.")


def latex_escape(text: object) -> str:
    s = str(text)
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def write_tables(summary_rows: list[dict], runs_dir: Path, table_dir: Path) -> None:
    table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        (table_dir / "final_results.tex").write_text(
            "\\begin{tabular}{ll}\\toprule Result & Status \\\\\\midrule "
            "No completed runs & This experiment was not run yet. \\\\\\bottomrule\\end{tabular}\n"
        )
    else:
        adamw = summary[summary["routing"] == "adamw_all"].groupby("dataset")["final_val_loss"].mean().to_dict()
        grouped = summary.groupby(["dataset", "routing"]).agg(
            mean_val_loss=("final_val_loss", "mean"),
            std_val_loss=("final_val_loss", "std"),
            seeds=("seed", "nunique"),
            best_checkpoint_step=("best_checkpoint_step", "max"),
            tokens_seen=("tokens_seen", "max"),
            wall_clock_seconds=("wall_clock_seconds", "mean"),
            tokens_per_sec=("tokens_per_sec", "mean"),
            peak_memory=("peak_memory", "max"),
        ).reset_index()
        grouped["improvement_over_adamw"] = grouped.apply(
            lambda r: adamw.get(r["dataset"], np.nan) - r["mean_val_loss"], axis=1
        )
        cols = [
            "dataset",
            "routing",
            "mean_val_loss",
            "std_val_loss",
            "seeds",
            "best_checkpoint_step",
            "tokens_seen",
            "wall_clock_seconds",
            "tokens_per_sec",
            "peak_memory",
            "improvement_over_adamw",
        ]
        write_df_table(grouped[cols], table_dir / "final_results.tex")

    group_rows = []
    for path in runs_dir.glob("*/optimizer_groups.json"):
        try:
            data = json.loads(path.read_text())
            group_rows.append(
                {
                    "run": path.parent.name,
                    "routing": data.get("routing", ""),
                    "muon_tensors": len(data.get("muon", [])),
                    "adamw_tensors": len(data.get("adamw", [])),
                }
            )
        except Exception:
            pass
    write_df_table(pd.DataFrame(group_rows), table_dir / "optimizer_group_counts.tex")
    write_df_table(pd.DataFrame(group_rows), table_dir / "routing_summary.tex")

    hp = pd.DataFrame(
        [
            {"hyperparameter": "AdamW learning rate", "value": "3e-4 default"},
            {"hyperparameter": "AdamW betas", "value": "(0.9, 0.95) default"},
            {"hyperparameter": "AdamW weight decay", "value": "0.1 default"},
            {"hyperparameter": "Muon learning rate", "value": "0.02 default"},
            {"hyperparameter": "Muon momentum", "value": "0.95 default"},
            {"hyperparameter": "Muon weight decay", "value": "0.05 default"},
        ]
    )
    write_df_table(hp, table_dir / "hyperparameters.tex")


def write_df_table(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        path.write_text(
            "\\begin{tabular}{ll}\\toprule Item & Status \\\\\\midrule "
            "No data & This experiment was not run yet. \\\\\\bottomrule\\end{tabular}\n"
        )
        return
    safe = df.copy()
    for col in safe.columns:
        safe[col] = safe[col].map(latex_escape)
    path.write_text(safe.to_latex(index=False, escape=False, longtable=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    report_dir = Path(args.report_dir)
    fig_dir = report_dir / "figures"
    table_dir = report_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    metrics, geometry, summaries = load_runs(runs_dir)
    summary = pd.DataFrame(summaries)
    line_plot(metrics, "tinystories", fig_dir / "validation_loss_vs_tokens_tinystories.png")
    line_plot(metrics, "fineweb_edu_10bt", fig_dir / "validation_loss_vs_tokens_finewebedu.png")
    wallclock_plot(metrics, fig_dir / "validation_loss_vs_wallclock.png")
    final_bar(summary, fig_dir / "final_loss_barplot.png")
    heatmap(summary, fig_dir / "routing_heatmap.png", layerwise=False)
    heatmap(summary, fig_dir / "layerwise_heatmap.png", layerwise=True)
    geometry_plot(geometry, "effective_rank", fig_dir / "update_effective_rank_by_module.png")
    geometry_plot(geometry, "update_norm", fig_dir / "update_norm_by_module.png")
    simple_placeholder_set(fig_dir)
    write_tables(summaries, runs_dir, table_dir)
    print(f"Wrote analysis assets to {report_dir}")


if __name__ == "__main__":
    main()
