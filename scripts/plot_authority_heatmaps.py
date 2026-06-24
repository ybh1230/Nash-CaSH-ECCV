import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

if os.name == "nt" and "WINDIR" not in os.environ:
    os.environ["WINDIR"] = r"C:\Windows"

import matplotlib.pyplot as plt
import torch


def load_record(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def stack_trimmed(vectors):
    vectors = [v.flatten().float() for v in vectors if v is not None and v.numel() > 0]
    if not vectors:
        return None
    length = min(v.numel() for v in vectors)
    return torch.stack([v[:length] for v in vectors], dim=0)


def write_summary(records, output_dir: Path):
    rows = []
    for rec in records:
        rows.append(
            {
                "file": rec["file"],
                "step": rec.get("step"),
                "call_index": rec.get("call_index"),
                "layer": rec.get("layer"),
                "mean": rec.get("mean"),
                "std": rec.get("std"),
                "min": rec.get("min"),
                "max": rec.get("max"),
                "shape": rec.get("shape"),
            }
        )

    with (output_dir / "authority_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "step", "call_index", "layer", "mean", "std", "min", "max", "shape"],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_step_token(records, output_dir: Path, max_rows: int):
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        key = rec.get("step")
        if key is None:
            key = rec.get("call_index", idx)
        groups[int(key)].append(rec.get("token_trace"))

    rows = []
    labels = []
    for key in sorted(groups.keys()):
        stacked = stack_trimmed(groups[key])
        if stacked is None:
            continue
        rows.append(stacked.mean(dim=0))
        labels.append(key)

    if not rows:
        return

    if len(rows) > max_rows:
        keep = torch.linspace(0, len(rows) - 1, steps=max_rows).round().long().tolist()
        rows = [rows[i] for i in keep]
        labels = [labels[i] for i in keep]

    matrix = stack_trimmed(rows)
    if matrix is None:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    im = ax.imshow(matrix.numpy(), aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title("Semantic Authority a(i): Denoising Step x Sampled Token")
    ax.set_xlabel("sampled token index")
    ax.set_ylabel("denoising step")
    if len(labels) <= 16:
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
    else:
        ticks = torch.linspace(0, len(labels) - 1, steps=10).round().long().tolist()
        ax.set_yticks(ticks)
        ax.set_yticklabels([labels[i] for i in ticks])
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label("contextual authority")
    fig.tight_layout()
    fig.savefig(output_dir / "authority_step_token_heatmap.png", dpi=220)
    plt.close(fig)


def plot_spatial_montage(records, output_dir: Path, max_panels: int):
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        spatial = rec.get("spatial_map")
        if spatial is None:
            continue
        key = rec.get("step")
        if key is None:
            key = rec.get("call_index", idx)
        groups[int(key)].append(spatial.float())

    maps = []
    labels = []
    for key in sorted(groups.keys()):
        stacked = stack_trimmed([m.flatten() for m in groups[key]])
        if stacked is None:
            continue
        base_shape = groups[key][0].shape
        maps.append(stacked.mean(dim=0).reshape(base_shape))
        labels.append(key)

    if not maps:
        return

    if len(maps) > max_panels:
        keep = torch.linspace(0, len(maps) - 1, steps=max_panels).round().long().tolist()
        maps = [maps[i] for i in keep]
        labels = [labels[i] for i in keep]

    cols = min(3, len(maps))
    rows = (len(maps) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")

    last_im = None
    for ax, label, spatial in zip(axes.flat, labels, maps):
        last_im = ax.imshow(spatial.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"step {label}")
        ax.axis("off")

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label("contextual authority")
    fig.suptitle("Spatial Semantic Authority Maps", y=0.98)
    fig.savefig(output_dir / "authority_spatial_montage.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mean_curve(records, output_dir: Path):
    groups = defaultdict(list)
    for idx, rec in enumerate(records):
        key = rec.get("step")
        if key is None:
            key = rec.get("call_index", idx)
        groups[int(key)].append(float(rec.get("mean", 0.0)))

    xs = []
    ys = []
    for key in sorted(groups.keys()):
        xs.append(key)
        ys.append(sum(groups[key]) / max(len(groups[key]), 1))

    if not xs:
        return

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.0)
    ax.set_title("Mean Contextual Authority Across Denoising")
    ax.set_xlabel("denoising step")
    ax.set_ylabel("mean a(i)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "authority_step_mean_curve.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot SemEq/Nash authority heatmaps.")
    parser.add_argument("--log_dir", required=True, help="Directory containing authority_*.pt files")
    parser.add_argument("--output_dir", default=None, help="Directory for plotted heatmaps")
    parser.add_argument("--max_rows", type=int, default=80, help="Maximum rows in the step-token heatmap")
    parser.add_argument("--max_spatial_panels", type=int, default=6, help="Maximum spatial maps in the montage")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir) if args.output_dir else log_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for path in sorted(log_dir.glob("authority_*.pt")):
        rec = load_record(path)
        rec["file"] = path.name
        records.append(rec)

    if not records:
        raise SystemExit(f"No authority_*.pt files found in {log_dir}")

    write_summary(records, output_dir)
    plot_step_token(records, output_dir, max_rows=args.max_rows)
    plot_spatial_montage(records, output_dir, max_panels=args.max_spatial_panels)
    plot_mean_curve(records, output_dir)
    print(f"Wrote authority plots to {output_dir}")


if __name__ == "__main__":
    main()
