#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import itertools
import warnings
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# -----------------------------
# Settings
# -----------------------------
DEFAULT_METRICS = ["train_c", "train_hr", "test_c", "test_hr", "val_c", "val_hr", "exval_c", "exval_hr"]

ARCH_ORDER = ["resnet", "effnet", "vit"]
WITHIN_ORDER = ["Baseline", "Top5", "Top10", "Top15", "Top20", "Top25"]

ARCH_BASE_COLORS = {
    "resnet": "#d62728",   # red
    "effnet": "#ff7f0e",   # orange
    "vit":    "#9467bd",   # purple
}

GROUP_SIZE = 6
GAP = 1

# -----------------------------
# I/O helpers
# -----------------------------
def read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def check_exact_six(label: str, items: List[str]):
    if len(items) != 6:
        raise ValueError(f"--{label} must have exactly 6 files in order: {', '.join(WITHIN_ORDER)}. Got {len(items)}.")

# -----------------------------
# Coloring
# -----------------------------
def hex_to_rgb01(hexcolor: str):
    return matplotlib.colors.to_rgb(hexcolor)

def rgb01_to_hex(rgb):
    return matplotlib.colors.to_hex(rgb)

def darken_by_features(base_hex: str, nf: int) -> str:
    rgb = np.array(hex_to_rgb01(base_hex))
    nf = max(5, min(25, int(nf)))
    factor = (nf - 5) / (25 - 5)
    out = (1 - factor) * rgb + factor * (0.2 * rgb)
    out = np.clip(out, 0, 1)
    return rgb01_to_hex(tuple(out))

def label_for(arch: str, within: str) -> str:
    pretty_arch = {"resnet":"ResNet", "effnet":"EfficientNet", "vit":"ViT"}[arch]
    return f"{pretty_arch} {within}"

def expected_feature_count(within: str) -> int:
    if within == "Baseline":
        return 0
    return int(within.replace("Top", ""))

# -----------------------------
# Data assembly
# -----------------------------
def assemble_long_form(arch_to_paths: Dict[str, List[str]],
                       metrics: List[str]) -> Tuple[Dict[str, List[pd.Series]], List[str], List[str], List[int]]:
    column_data = {m: [] for m in metrics}
    labels, arch_tags, feat_counts = [], [], []

    for arch in ARCH_ORDER:
        paths = arch_to_paths[arch]
        check_exact_six(arch, paths)

        for within, path in zip(WITHIN_ORDER, paths):
            df = read_table(path)
            for m in metrics:
                if m in df.columns and pd.api.types.is_numeric_dtype(df[m]):
                    col = pd.to_numeric(df[m], errors="coerce").dropna()
                    column_data[m].append(col)
                else:
                    column_data[m].append(pd.Series(dtype=float))
            labels.append(label_for(arch, within))
            arch_tags.append(arch)
            feat_counts.append(expected_feature_count(within))

    return column_data, labels, arch_tags, feat_counts

# -----------------------------
# Plotting (grouped with gaps)
# -----------------------------
def y_limits_for(metric_key: str):
    is_train = metric_key.startswith("train")
    if metric_key.endswith("_hr"):
        return (-0.5, 23) if not is_train else None
    if metric_key.endswith("_c"):
        # For C-index, cap non-training plots at 0.90 as requested
        return (0.40, 0.90) if not is_train else None
    return None

def grouped_positions() -> List[float]:
    positions = []
    start = 1
    for _ in ARCH_ORDER:
        positions.extend(list(range(start, start + GROUP_SIZE)))
        start += GROUP_SIZE + GAP
    return positions

def make_boxplots_grouped(column_data: Dict[str, List[pd.Series]],
                          labels: List[str],
                          arch_tags: List[str],
                          feat_counts: List[int],
                          outdir: str):
    facecolors, edgecolors = [], []
    for arch, nf, lab in zip(arch_tags, feat_counts, labels):
        base = ARCH_BASE_COLORS[arch]
        if nf == 0:
            facecolors.append("#FFFFFF")
            edgecolors.append(base)
        else:
            facecolors.append(darken_by_features(base, nf))
            edgecolors.append(base)

    positions = grouped_positions()

    for metric, series_list in column_data.items():
        # Divide test_hr by 2 before plotting, leave stats untouched
        if metric == "test_hr":
            data_arrays = [ (s.values / 2.0) for s in series_list ]
        else:
            data_arrays = [ s.values for s in series_list ]
        fig, ax = plt.subplots(figsize=(12, 6))
        bp = ax.boxplot(
            data_arrays,
            positions=positions,
            widths=0.7,
            patch_artist=True,
            showmeans=False,
            showfliers=False
        )

        for i, (box, med) in enumerate(zip(bp["boxes"], bp["medians"])):
            box.set_facecolor(facecolors[i])
            box.set_edgecolor(edgecolors[i])
            box.set_linewidth(2)
            med.set_color("black")
            med.set_linewidth(2)

        ax.set_title(metric, fontsize=16)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xticks([])

        # (Removed gray background spans between groups as requested)

        # (Legend removed as requested)

        ylim = y_limits_for(metric)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Per-box tick labels: feature counts (blank for baselines)
        feat_ticklabels = ["" if nf == 0 else str(nf) for nf in feat_counts]
        ax.set_xticks(positions)
        ax.set_xticklabels(feat_ticklabels, fontsize=11)

        # Add model superlabels under the x-axis
        centers = [np.mean(positions[i*(GROUP_SIZE+GAP): i*(GROUP_SIZE+GAP)+GROUP_SIZE]) for i in range(3)]
        for x_center, name in zip(centers, ["ResNet", "EfficientNet", "ViT"]):
            ax.text(x_center, -0.10, name, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=12)

        os.makedirs(outdir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"{metric}.png"), dpi=300)
        fig.savefig(os.path.join(outdir, f"{metric}.pdf"))
        plt.close(fig)

# -----------------------------
# Statistics
# -----------------------------
def compute_and_save_statistics_all(column_data: Dict[str, List[pd.Series]],
                                    labels: List[str],
                                    out_xlsx: str):
    with pd.ExcelWriter(out_xlsx) as writer:
        for i, lab in enumerate(labels):
            rows = []
            for m, series_list in column_data.items():
                col = series_list[i]
                if len(col) == 0:
                    continue
                mean_val = float(np.mean(col))
                med_val = float(np.median(col))
                std_val = float(np.std(col, ddof=1)) if len(col) > 1 else 0.0
                rows.append({
                    "Column": m,
                    "Mean": mean_val,
                    "Median": med_val,
                    "Standard Deviation": std_val,
                    "Mean ± Std Dev": f"{mean_val:.2f} ± {std_val:.2f}"
                })
            pd.DataFrame(rows).to_excel(writer, sheet_name=lab[:31], index=False)

        anova_rows, tukey_frames, ftest_rows = [], [], []
        for m, groups in column_data.items():
            valid = [(g, labels[idx]) for idx, g in enumerate(groups) if len(g) > 0]
            if len(valid) < 2:
                continue
            data_only = [g for g, _ in valid]
            try:
                f_stat, p_val = stats.f_oneway(*data_only)
            except Exception:
                f_stat, p_val = np.nan, np.nan
            anova_rows.append({"Column": m, "F-statistic": f_stat, "p-value": p_val})

            if pd.notna(p_val) and p_val < 0.05:
                combined = pd.concat(data_only, ignore_index=True)
                group_labels = []
                for g, lbl in valid:
                    group_labels.extend([lbl] * len(g))
                tuk = pairwise_tukeyhsd(endog=combined.values,
                                        groups=np.array(group_labels),
                                        alpha=0.05)
                tdf = pd.DataFrame(tuk.summary().data[1:], columns=tuk.summary().data[0])
                tdf.insert(0, "Column", m)
                tukey_frames.append(tdf)

            for (g1, l1), (g2, l2) in itertools.combinations(valid, 2):
                try:
                    f_stat, p_v = stats.levene(g1.values, g2.values)
                except Exception:
                    f_stat, p_v = np.nan, np.nan
                ftest_rows.append({
                    "Column": m,
                    "Group 1": l1,
                    "Group 2": l2,
                    "F-statistic": f_stat,
                    "p-value": p_v
                })

        if anova_rows:
            pd.DataFrame(anova_rows).to_excel(writer, sheet_name="ANOVA Results", index=False)
        if tukey_frames:
            pd.concat(tukey_frames, ignore_index=True).to_excel(writer, sheet_name="Tukey HSD Results", index=False)
        if ftest_rows:
            pd.DataFrame(ftest_rows).to_excel(writer, sheet_name="Pairwise Levene", index=False)

# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Grouped ResNet/EfficientNet/ViT stats + plots (6 per arch)")
    p.add_argument("--resnet", nargs=6, required=True)
    p.add_argument("--effnet", nargs=6, required=True)
    p.add_argument("--vit", nargs=6, required=True)
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--outdir", required=True)
    p.add_argument("--outfile", default="radpre_grouped_stats.xlsx")
    args = p.parse_args()

    arch_to_paths = {"resnet": args.resnet, "effnet": args.effnet, "vit": args.vit}
    column_data, labels, arch_tags, feat_counts = assemble_long_form(arch_to_paths, args.metrics)
    make_boxplots_grouped(column_data, labels, arch_tags, feat_counts, args.outdir)

    out_xlsx = os.path.join(args.outdir, args.outfile)
    compute_and_save_statistics_all(column_data, labels, out_xlsx)
    print(f"Statistics saved to {out_xlsx}")

if __name__ == "__main__":
    main()

