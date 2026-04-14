#!/usr/bin/env python3
"""
Visualize flat predgeom context decisions from encoder CSV dumps.

Example:
  python3 scripts/visualize_predgeom_contexts.py \
    --ply input.ply \
    --dump-csv context_dump.csv \
    --out-dir scripts/out_context_viz
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "coded_idx",
    "src_idx",
    "laser_idx",
    "ctx_group",
    "mode_range_class",
    "mode_boundary",
    "mode",
    "res_l1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize predgeom context effects")
    parser.add_argument("--ply", required=True, help="Input point cloud PLY path")
    parser.add_argument("--dump-csv", required=True, help="CSV path dumped by encoder")
    parser.add_argument(
        "--out-dir",
        default="scripts/out_context_viz",
        help="Directory for generated plots and summary tables",
    )
    parser.add_argument(
        "--no-open3d",
        action="store_true",
        help="Disable interactive Open3D views",
    )
    return parser.parse_args()


def check_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")


def entropy_from_counts(counts: np.ndarray) -> float:
    total = float(np.sum(counts))
    if total <= 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def save_occupancy_plot(df: pd.DataFrame, out_dir: Path) -> None:
    table = (
        df.groupby(["ctx_group", "mode_range_class", "mode_boundary"])
        .size()
        .reset_index(name="count")
    )
    table["ctx"] = (
        "g"
        + table["ctx_group"].astype(str)
        + "_r"
        + table["mode_range_class"].astype(str)
        + "_b"
        + table["mode_boundary"].astype(str)
    )
    table = table.sort_values(["ctx_group", "mode_range_class", "mode_boundary"])

    plt.figure(figsize=(12, 4))
    plt.bar(table["ctx"], table["count"])
    plt.title("Context Occupancy: (group, rangeClass, boundary)")
    plt.xlabel("Context Bin")
    plt.ylabel("Point Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "occupancy_by_context.png", dpi=180)
    plt.close()


def save_mode_distribution_plots(df: pd.DataFrame, out_dir: Path) -> None:
    context_mode = (
        df.groupby(["ctx_group", "mode_range_class", "mode_boundary", "mode"])
        .size()
        .reset_index(name="count")
    )
    context_mode["ctx"] = (
        "g"
        + context_mode["ctx_group"].astype(str)
        + "_r"
        + context_mode["mode_range_class"].astype(str)
        + "_b"
        + context_mode["mode_boundary"].astype(str)
    )
    pivot = context_mode.pivot(index="ctx", columns="mode", values="count").fillna(0)
    pivot = pivot.sort_index()

    ax = pivot.plot(kind="bar", stacked=True, figsize=(13, 5), colormap="tab20")
    ax.set_title("Mode Distribution by Context")
    ax.set_xlabel("Context Bin")
    ax.set_ylabel("Point Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "mode_distribution_by_context.png", dpi=180)
    plt.close()

    boundary_mode = (
        df.groupby(["mode_boundary", "mode"]).size().reset_index(name="count")
    )
    pivot2 = boundary_mode.pivot(index="mode_boundary", columns="mode", values="count").fillna(0)
    ax = pivot2.plot(kind="bar", stacked=True, figsize=(8, 4), colormap="tab20")
    ax.set_title("Mode Distribution: boundary=0 vs 1")
    ax.set_xlabel("mode_boundary")
    ax.set_ylabel("Point Count")
    plt.tight_layout()
    plt.savefig(out_dir / "mode_distribution_boundary_split.png", dpi=180)
    plt.close()


def save_residual_plots(df: pd.DataFrame, out_dir: Path) -> None:
    grouped = (
        df.groupby(["ctx_group", "mode_range_class", "mode_boundary"])["res_l1"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
    )
    grouped["ctx"] = (
        "g"
        + grouped["ctx_group"].astype(str)
        + "_r"
        + grouped["mode_range_class"].astype(str)
        + "_b"
        + grouped["mode_boundary"].astype(str)
    )
    grouped = grouped.sort_values(["ctx_group", "mode_range_class", "mode_boundary"])

    plt.figure(figsize=(12, 4))
    plt.bar(grouped["ctx"], grouped["mean"])
    plt.title("Average Residual L1 by Context")
    plt.xlabel("Context Bin")
    plt.ylabel("Mean L1")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_l1_by_context.png", dpi=180)
    plt.close()

    grouped.to_csv(out_dir / "residual_stats_by_context.csv", index=False)


def save_summary_table(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for (g, rc, b), sub in df.groupby(["ctx_group", "mode_range_class", "mode_boundary"]):
        mode_counts = (
            sub.groupby("mode").size().reindex(range(6), fill_value=0).to_numpy(dtype=np.float64)
        )
        rows.append(
            {
                "ctx_group": g,
                "mode_range_class": rc,
                "mode_boundary": b,
                "count": int(len(sub)),
                "mode_entropy_bits": entropy_from_counts(mode_counts),
                "mean_res_l1": float(sub["res_l1"].mean()),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["ctx_group", "mode_range_class", "mode_boundary"]
    )
    summary.to_csv(out_dir / "context_summary.csv", index=False)


def _colors_from_label(values: np.ndarray, palette: np.ndarray) -> np.ndarray:
    colors = np.zeros((values.shape[0], 3), dtype=np.float64)
    for idx, color in enumerate(palette):
        colors[values == idx] = color
    return colors


def run_open3d_views(df: pd.DataFrame, ply_path: Path) -> None:
    try:
        import open3d as o3d
    except ImportError:
        print("[warn] open3d not installed. Skipping interactive 3D views.")
        return

    pcd = o3d.io.read_point_cloud(str(ply_path))
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        print("[warn] Empty PLY point cloud. Skipping interactive 3D views.")
        return

    # Use src_idx mapping from dump to color point cloud in original input order.
    if df["src_idx"].max() >= points.shape[0]:
        print("[warn] src_idx exceeds PLY point count. Skipping interactive 3D views.")
        return

    group_values = np.zeros(points.shape[0], dtype=np.int32)
    boundary_values = np.zeros(points.shape[0], dtype=np.int32)
    range_values = np.zeros(points.shape[0], dtype=np.int32)
    mode_values = np.zeros(points.shape[0], dtype=np.int32)

    src_idx = df["src_idx"].to_numpy(dtype=np.int64)
    group_values[src_idx] = df["ctx_group"].to_numpy(dtype=np.int32)
    boundary_values[src_idx] = df["mode_boundary"].to_numpy(dtype=np.int32)
    range_values[src_idx] = df["mode_range_class"].to_numpy(dtype=np.int32)
    mode_values[src_idx] = df["mode"].to_numpy(dtype=np.int32)

    palettes = {
        "group": np.array([[0.2, 0.6, 1.0], [1.0, 0.45, 0.2]]),
        "boundary": np.array([[0.65, 0.65, 0.65], [1.0, 0.0, 0.0]]),
        "range": np.array([[0.2, 0.7, 0.3], [0.95, 0.8, 0.1], [0.6, 0.2, 0.9]]),
        "mode": np.array(
            [
                [0.6, 0.6, 0.6],
                [0.1, 0.7, 0.95],
                [0.2, 0.6, 0.2],
                [0.9, 0.4, 0.1],
                [0.8, 0.2, 0.7],
                [0.95, 0.9, 0.2],
            ]
        ),
    }

    views = [
        ("Context Group", group_values, palettes["group"]),
        ("Boundary", boundary_values, palettes["boundary"]),
        ("Range Class", range_values, palettes["range"]),
        ("Mode", mode_values, palettes["mode"]),
    ]

    for title, values, palette in views:
        colored = o3d.geometry.PointCloud()
        colored.points = o3d.utility.Vector3dVector(points)
        colored.colors = o3d.utility.Vector3dVector(_colors_from_label(values, palette))
        print(f"[open3d] Showing: {title}")
        o3d.visualization.draw_geometries([colored], window_name=f"PredGeom {title}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.dump_csv)
    check_columns(df)

    # Ensure types for stable grouping / plotting.
    for col in ["ctx_group", "mode_range_class", "mode_boundary", "mode"]:
        df[col] = df[col].astype(int)
    df["res_l1"] = df["res_l1"].astype(float)

    save_occupancy_plot(df, out_dir)
    save_mode_distribution_plots(df, out_dir)
    save_residual_plots(df, out_dir)
    save_summary_table(df, out_dir)

    print(f"[done] Plots and tables written to: {out_dir}")

    if not args.no_open3d:
        run_open3d_views(df, Path(args.ply))


if __name__ == "__main__":
    main()
