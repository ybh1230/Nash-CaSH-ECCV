import argparse
import csv
import glob
import json
from pathlib import Path


def scalar_metric(value):
    if isinstance(value, list) and value:
        return value[0]
    if isinstance(value, (int, float)):
        return value
    return None


def main():
    parser = argparse.ArgumentParser(description="Summarize VBench eval_results JSON files.")
    parser.add_argument("--results_dir", default="vbench_results")
    parser.add_argument("--out_csv", default="vbench_results/summary.csv")
    parser.add_argument("--out_md", default="vbench_results/summary.md")
    args = parser.parse_args()

    rows = []
    for path in sorted(glob.glob(str(Path(args.results_dir) / "*eval_results.json"))):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for metric, value in data.items():
            score = scalar_metric(value)
            details = value[1] if isinstance(value, list) and len(value) > 1 else None
            rows.append({
                "file": Path(path).name,
                "metric": metric,
                "score": score,
                "score_percent": None if score is None else score * 100.0,
                "details": json.dumps(details, ensure_ascii=False),
            })

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "score", "score_percent", "file", "details"])
        writer.writeheader()
        writer.writerows(rows)

    with out_md.open("w", encoding="utf-8") as f:
        f.write("| Metric | Score | Percent | Source File |\n")
        f.write("|---|---:|---:|---|\n")
        for row in rows:
            score = "" if row["score"] is None else f"{row['score']:.6f}"
            percent = "" if row["score_percent"] is None else f"{row['score_percent']:.2f}%"
            f.write(f"| {row['metric']} | {score} | {percent} | {row['file']} |\n")

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
