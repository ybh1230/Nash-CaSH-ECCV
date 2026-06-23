# Nash-CaSH Visualization Protocol

This protocol is for qualitative figures and small supporting measurements. It is not a replacement for the 60-prompt VBench table.

## Goal

The visualization section should demonstrate the Nash semantic-authority idea without adding a large amount of extra generation. We compare:

- `fixed`: fixed authority allocation using `--no-nash_cash`.
- `nash_cash`: Nash semantic authority allocation using the default Nash-CaSH settings.

Both methods must use the same prompt, seed, resolution, cache interval, and model version.

## Prompt Suite

Use `experiments/visual_prompts.json`. The suite contains 8 prompts selected from the provided figure/case-study prompts:

- face close-up and texture
- armored animal identity
- library repeated structures
- river reflections and small petals
- desert layout and small subjects
- hummingbird fine motion
- jellyfish translucent detail
- motorcycle specular reflections

This suite is intended for visual evidence. The final main table should use the 60 VBench prompts when they are available.

## Recommended First Run

For a fast but useful comparison:

```bash
python scripts/run_visual_suite.py \
  --prompts experiments/visual_prompts.json \
  --output_dir outputs/visual_suite \
  --methods nash_cash fixed \
  --target_height 720 \
  --target_width 1280 \
  --cache_steps 2 \
  --seed 2026 \
  --max_prompts 4
```

For final paper figures, rerun the strongest 3 or 4 prompts at 1080P:

```bash
python scripts/run_visual_suite.py \
  --prompts experiments/visual_prompts.json \
  --output_dir outputs/visual_suite_1080p \
  --methods nash_cash fixed \
  --target_height 1088 \
  --target_width 1920 \
  --cache_steps 2 \
  --seed 2026 \
  --ids fig2_elderly_closeup fig5_armored_wolf fig10_ancient_library fig10_rainy_motorcycle
```

## Extract Frames and Crops

After generation:

```bash
python scripts/extract_visual_evidence.py \
  --prompts experiments/visual_prompts.json \
  --video_root outputs/visual_suite \
  --methods fixed nash_cash \
  --out_dir outputs/visual_evidence
```

The script saves:

- `outputs/visual_evidence/frames/`: individual labeled frames.
- `outputs/visual_evidence/crops/`: center crops for local detail.
- `outputs/visual_evidence/sheets/`: comparison sheets for each prompt.
- `outputs/visual_evidence/metadata.csv`: frame count, FPS, and resolution.

## Optional VBench Support

For visual-suite videos, run the six custom-input metrics one dimension at a time:

```bash
for DIM in subject_consistency background_consistency motion_smoothness dynamic_degree aesthetic_quality imaging_quality
do
  vbench evaluate \
    --videos_path outputs/visual_suite/nash_cash \
    --mode custom_input \
    --dimension "$DIM" \
    --output_path vbench_visual_results
done
```

Summarize the JSON results:

```bash
python scripts/summarize_vbench_results.py \
  --results_dir vbench_visual_results \
  --out_csv vbench_visual_results/summary.csv \
  --out_md vbench_visual_results/summary.md
```

## Reporting Language

Use cautious wording:

- Correct: "We use the figure/case-study prompts for qualitative analysis."
- Correct: "The small VBench custom-input metrics are supporting evidence."
- Incorrect: "These results reproduce the main 60-prompt VBench table."

When the 60 VBench prompts are available, use them for the official table and keep this visualization suite for qualitative figures and appendix evidence.
