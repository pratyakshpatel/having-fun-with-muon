#!/usr/bin/env python3
"""Generate LaTeX assets that are easier to build from run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--report_dir", default="report")
    return p.parse_args()


def esc(value: object, limit: int = 240) -> str:
    text = str(value)
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("#", "\\#")
        .replace("$", "\\$")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def load_samples(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for sample_path in sorted(runs_dir.glob("*/samples.jsonl")):
        for line in sample_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                row["run"] = sample_path.parent.name
                rows.append(row)
            except json.JSONDecodeError:
                pass
    return pd.DataFrame(rows)


def qualitative_table(samples: pd.DataFrame, path: Path) -> None:
    if samples.empty:
        path.write_text(
            "\\begin{longtable}{p{0.22\\linewidth}p{0.68\\linewidth}}\\toprule "
            "Prompt & Status \\\\\\midrule "
            "Qualitative samples & Qualitative samples were not generated for this run. "
            "\\\\\\bottomrule\\end{longtable}\n",
            encoding="utf-8",
        )
        return
    latest = samples.sort_values("step").groupby(["prompt", "routing"], as_index=False).tail(1)
    rows = []
    for prompt, group in latest.groupby("prompt"):
        adamw = group[group["routing"] == "adamw_all"]
        all_hidden = group[group["routing"] == "muon_all_hidden"]
        routed = group[~group["routing"].isin(["adamw_all", "muon_all_hidden"])]
        best = routed.iloc[-1:] if not routed.empty else pd.DataFrame()
        rows.append(
            {
                "Prompt": esc(prompt, 160),
                "AdamW output": esc(adamw.iloc[-1]["generated_text"] if not adamw.empty else "Not available."),
                "Muon all hidden output": esc(
                    all_hidden.iloc[-1]["generated_text"] if not all_hidden.empty else "Not available."
                ),
                "Best routed Muon output": esc(best.iloc[-1]["generated_text"] if not best.empty else "Not available."),
                "My observation": "Not assessed automatically.",
            }
        )
    df = pd.DataFrame(rows)
    path.write_text(df.to_latex(index=False, escape=False, longtable=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report_dir = Path(args.report_dir)
    table_dir = report_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    qualitative_table(load_samples(Path(args.runs_dir)), table_dir / "qualitative_samples.tex")
    print(f"Wrote report assets to {report_dir}")


if __name__ == "__main__":
    main()
