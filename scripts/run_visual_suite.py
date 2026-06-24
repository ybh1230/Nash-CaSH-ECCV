import argparse
import json
import subprocess
import sys
from pathlib import Path


def load_prompts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        prompts = json.load(f)
    for item in prompts:
        if "id" not in item or "prompt" not in item:
            raise ValueError(f"Each prompt item needs id and prompt fields: {item}")
    return prompts


def write_prompt_file(prompt_dir: Path, prompt_id: str, prompt: str) -> Path:
    prompt_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = prompt_dir / f"{prompt_id}.txt"
    prompt_path.write_text(prompt.strip() + "\n", encoding="utf-8")
    return prompt_path


def run_one(args, prompt_item, method_name):
    output_dir = Path(args.output_dir) / method_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{prompt_item['id']}.mp4"
    if output_path.exists() and not args.overwrite:
        print(f"[skip] {output_path}")
        return

    prompt_path = write_prompt_file(Path(args.output_dir) / "prompts", prompt_item["id"], prompt_item["prompt"])
    cmd = [
        sys.executable,
        "inference.py",
        "--mode",
        args.mode,
        "--target_height",
        str(args.target_height),
        "--target_width",
        str(args.target_width),
        "--cache_steps",
        str(args.cache_steps),
        "--seed",
        str(args.seed),
        "--prompt_file",
        str(prompt_path),
        "--output",
        str(output_path),
        "--route_mode",
        method_name,
    ]
    if args.log_maps:
        log_dir = Path(args.output_dir) / "logs" / method_name / prompt_item["id"]
        cmd += [
            "--authority_log_dir",
            str(log_dir),
            "--authority_log_stride",
            str(args.authority_log_stride),
            "--authority_log_token_stride",
            str(args.authority_log_token_stride),
        ]
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the qualitative SemEq visual suite.")
    parser.add_argument("--prompts", default="experiments/visual_prompts.json")
    parser.add_argument("--output_dir", default="outputs/visual_suite")
    parser.add_argument("--methods", nargs="+", default=["full"], choices=["original", "hafp", "nash", "full"], help="Ablation methods to run")
    parser.add_argument("--ids", nargs="*", default=None, help="Optional prompt ids to run")
    parser.add_argument("--max_prompts", type=int, default=None, help="Optional maximum number of prompts to run")
    parser.add_argument("--mode", choices=["cache", "nocache"], default="cache")
    parser.add_argument("--target_height", type=int, default=720)
    parser.add_argument("--target_width", type=int, default=1280)
    parser.add_argument("--cache_steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--log_maps", action="store_true", help="Save compressed authority and purification maps")
    parser.add_argument("--authority_log_stride", type=int, default=4)
    parser.add_argument("--authority_log_token_stride", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    prompts = load_prompts(Path(args.prompts))
    if args.ids:
        wanted = set(args.ids)
        prompts = [item for item in prompts if item["id"] in wanted]
    if args.max_prompts is not None:
        prompts = prompts[: args.max_prompts]
    for item in prompts:
        for method_name in args.methods:
            run_one(args, item, method_name)


if __name__ == "__main__":
    main()
