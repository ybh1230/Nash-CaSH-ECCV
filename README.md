# Nash-CaSH

Game-Theoretic Cross-Attention Harmonization for Training-Free Ultra-High-Resolution Video Generation.

Nash-CaSH is a training-free high-resolution video generation method. It combines native-scale local attention with global semantic guidance, then uses a Nash social-welfare gate to decide how much semantic authority each branch should receive at every cross-attention token.

## What Is New

- Token-wise Nash equilibrium gate for local-global cross-attention fusion.
- Training-free inference with no finetuning or extra model parameters.
- Cache-compatible global branch reuse.
- Built-in ablation switches for fixed override, Nash bounds, branch priors, and cache intervals.
- ECCV-style paper draft and experiment protocol included under `paper/` and `experiments/`.

## Files

- `nash_cash.py`: Nash payoff and equilibrium fusion.
- `inference.py`: end-to-end generation entry point.
- `cache/`: cached high-resolution inference path.
- `nocache/`: non-cached high-resolution inference path.
- `scripts/run_nash_cash_ablation.ps1`: ablation runner.
- `scripts/run_visual_suite.py`: qualitative prompt-suite runner for Nash-CaSH and fixed baseline.
- `scripts/extract_visual_evidence.py`: frame/crop/contact-sheet extractor for paper figures.
- `scripts/summarize_vbench_results.py`: helper for aggregating VBench JSON outputs.
- `experiments/nash_cash_protocol.md`: evaluation protocol.
- `experiments/visual_prompts.json`: 8-prompt qualitative suite from the provided figure/case-study prompts.
- `experiments/visualization_protocol.md`: qualitative figure and supporting-metric protocol.
- `paper/nash_cash_eccv_revision.md`: full paper draft in Markdown.
- `paper/nash_cash_eccv_revision.tex`: LaTeX paper draft.

## Installation

Use the same dependency versions required by the Wan Diffusers pipeline and FlexAttention:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.35.2 opencv-python
```

## Run

Default Nash-CaSH cached inference:

```bash
python inference.py --mode cache --target_height 1088 --target_width 1920 --cache_steps 2 --output Nash_CaSH.mp4
```

Non-cached inference:

```bash
python inference.py --mode nocache --target_height 1088 --target_width 1920 --output Nash_CaSH_nocache.mp4
```

Fixed local-global override baseline:

```bash
python inference.py --mode cache --target_height 1088 --target_width 1920 --cache_steps 2 --no-nash_cash --output fixed_override.mp4
```

## Nash Parameters

```bash
python inference.py ^
  --mode cache ^
  --nash_floor 0.65 ^
  --nash_ceiling 0.98 ^
  --nash_full_prior 1.25 ^
  --nash_window_prior 1.0 ^
  --nash_temperature 1.0
```

## Ablation

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_nash_cash_ablation.ps1 -TargetHeight 1088 -TargetWidth 1920 -CacheSteps 2
```

## Evaluation

Follow `experiments/nash_cash_protocol.md` for official metric runs. Do not fill the paper's result tables until VBench and runtime evaluations are completed with fixed seeds.

## Qualitative Visualization

Use the lightweight prompt suite before the 60-prompt VBench table is ready:

```bash
python scripts/run_visual_suite.py --methods nash_cash fixed --target_height 720 --target_width 1280 --seed 2026 --max_prompts 4
python scripts/extract_visual_evidence.py --video_root outputs/visual_suite --methods fixed nash_cash
```

The generated contact sheets are saved under `outputs/visual_evidence/sheets/`. See `experiments/visualization_protocol.md` for the full workflow and reporting language.
