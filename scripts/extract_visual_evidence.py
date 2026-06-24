import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


FRAME_RATIOS = [0.08, 0.35, 0.62, 0.90]


def read_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_video_frames(path: Path, ratios):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    for ratio in ratios:
        idx = min(max(int(round((total - 1) * ratio)), 0), max(total - 1, 0))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot read frame {idx} from {path}")
        frames.append((idx, frame))
    cap.release()
    return frames, {"frames": total, "fps": fps, "width": width, "height": height}


def resize_to_height(image, height):
    h, w = image.shape[:2]
    if h == height:
        return image
    width = int(round(w * height / h))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def center_crop(image, frac=0.42):
    h, w = image.shape[:2]
    crop_w = max(64, int(w * frac))
    crop_h = max(64, int(h * frac))
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    return image[y0:y0 + crop_h, x0:x0 + crop_w]


def draw_label(image, label):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(out, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def stack_grid(rows, cell_height=220):
    normalized_rows = []
    for row in rows:
        cells = [resize_to_height(cell, cell_height) for cell in row]
        normalized_rows.append(np.concatenate(cells, axis=1))
    width = max(row.shape[1] for row in normalized_rows)
    padded = []
    for row in normalized_rows:
        if row.shape[1] < width:
            pad = np.full((row.shape[0], width - row.shape[1], 3), 255, dtype=np.uint8)
            row = np.concatenate([row, pad], axis=1)
        padded.append(row)
    return np.concatenate(padded, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract frames, crops, and contact sheets for qualitative evidence.")
    parser.add_argument("--prompts", default="experiments/visual_prompts.json")
    parser.add_argument("--video_root", default="outputs/visual_suite")
    parser.add_argument("--methods", nargs="+", default=["original", "hafp", "nash", "full"])
    parser.add_argument("--out_dir", default="outputs/visual_evidence")
    parser.add_argument("--cell_height", type=int, default=220)
    args = parser.parse_args()

    prompts = read_prompts(Path(args.prompts))
    video_root = Path(args.video_root)
    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    crop_dir = out_dir / "crops"
    sheet_dir = out_dir / "sheets"
    for d in (frame_dir, crop_dir, sheet_dir):
        d.mkdir(parents=True, exist_ok=True)

    metadata_rows = []
    for item in prompts:
        prompt_id = item["id"]
        frame_rows = []
        crop_rows = []
        for method in args.methods:
            video_path = video_root / method / f"{prompt_id}.mp4"
            if not video_path.exists():
                print(f"[missing] {video_path}")
                continue
            frames, meta = read_video_frames(video_path, FRAME_RATIOS)
            metadata_rows.append({
                "prompt_id": prompt_id,
                "source": item.get("source", ""),
                "focus": item.get("focus", ""),
                "method": method,
                "video_path": str(video_path),
                **meta,
            })

            frame_cells = []
            crop_cells = []
            for pos, (frame_idx, frame) in enumerate(frames):
                label = f"{method} | f={frame_idx}"
                labeled = draw_label(frame, label)
                crop = draw_label(center_crop(frame), label)
                frame_path = frame_dir / f"{prompt_id}_{method}_{pos:02d}.jpg"
                crop_path = crop_dir / f"{prompt_id}_{method}_{pos:02d}.jpg"
                cv2.imwrite(str(frame_path), labeled)
                cv2.imwrite(str(crop_path), crop)
                frame_cells.append(labeled)
                crop_cells.append(crop)
            frame_rows.append(frame_cells)
            crop_rows.append(crop_cells)

        if frame_rows:
            cv2.imwrite(str(sheet_dir / f"{prompt_id}_frames.jpg"), stack_grid(frame_rows, args.cell_height))
            cv2.imwrite(str(sheet_dir / f"{prompt_id}_crops.jpg"), stack_grid(crop_rows, args.cell_height))

    with (out_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["prompt_id", "source", "focus", "method", "video_path", "frames", "fps", "width", "height"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Saved visual evidence to {out_dir}")


if __name__ == "__main__":
    main()
