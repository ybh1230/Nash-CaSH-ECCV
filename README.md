# SemEq

Equilibrium-Guided Semantic Coordination for Training-Free High-Resolution Video Generation.

SemEq is a training-free high-resolution video generation method. It combines native-grid evidence with contextual semantic evidence, then uses a disagreement-aware Nash semantic coordination rule to decide how much authority each evidence source should receive at every cross-attention token.

## What Is New

- Token-wise disagreement-aware Nash semantic authority allocation.
- Training-free inference with no finetuning or extra model parameters.
- Cache-compatible contextual evidence reuse.
- Built-in controls for Nash bounds, evidence priors, authority momentum, cache intervals, and authority heatmaps.
- ECCV-style paper draft and experiment protocol included under `paper/` and `experiments/`.

## Files

- `nash_cash.py`: SemEq reliability estimation and Nash authority allocation.
- `inference.py`: end-to-end generation entry point.
- `cache/`: cached high-resolution inference path.
- `nocache/`: non-cached high-resolution inference path.
- `scripts/run_nash_cash_ablation.ps1`: method-variant runner for cache settings.
- `scripts/run_visual_suite.py`: qualitative prompt-suite runner for the full SemEq method.
- `scripts/extract_visual_evidence.py`: frame/crop/contact-sheet extractor for paper figures.
- `scripts/summarize_vbench_results.py`: helper for aggregating VBench JSON outputs.
- `scripts/plot_authority_heatmaps.py`: plots token/step and spatial heatmaps of the Nash authority map `a(i)`.
- `experiments/nash_cash_protocol.md`: evaluation protocol.
- `experiments/visual_prompts.json`: 8-prompt qualitative suite from the provided figure/case-study prompts.
- `experiments/visualization_protocol.md`: qualitative figure and supporting-metric protocol.
- `paper/main_eccv_overleaf.tex`: Overleaf-ready paper draft.

## Installation

Use the same dependency versions required by the Wan Diffusers pipeline and FlexAttention:

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.35.2 opencv-python
```

## Run

Default SemEq cached inference:

```bash
python inference.py --mode cache --target_height 1088 --target_width 1920 --cache_steps 2 --output SemEq.mp4
```

Non-cached inference:

```bash
python inference.py --mode nocache --target_height 1088 --target_width 1920 --output Nash_CaSH_nocache.mp4
```

## Nash Parameters

```bash
python inference.py ^
  --mode cache ^
  --nash_floor 0.65 ^
  --nash_ceiling 0.98 ^
  --nash_full_prior 1.25 ^
  --nash_window_prior 1.0 ^
  --nash_temperature 1.0 ^
  --nash_authority_momentum 0.10
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
python scripts/run_visual_suite.py --target_height 720 --target_width 1280 --seed 2026 --max_prompts 4
python scripts/extract_visual_evidence.py --video_root outputs/visual_suite --methods sem_eq
```

The generated contact sheets are saved under `outputs/visual_evidence/sheets/`. See `experiments/visualization_protocol.md` for the full workflow and reporting language.

## Authority Heatmaps

To visualize whether semantic authority changes across tokens and denoising steps, enable compressed authority logging during inference:

```bash
python inference.py \
  --mode cache \
  --target_height 1088 \
  --target_width 1920 \
  --cache_steps 2 \
  --seed 2026 \
  --authority_log_dir outputs/authority_logs/fig2_elderly \
  --authority_log_stride 4 \
  --authority_log_token_stride 256 \
  --output outputs/fig2_elderly_authority.mp4
```

Then render the heatmaps:

```bash
python scripts/plot_authority_heatmaps.py \
  --log_dir outputs/authority_logs/fig2_elderly \
  --output_dir outputs/authority_logs/fig2_elderly/plots
```

The key outputs are:

- `authority_step_token_heatmap.png`: denoising step by sampled token heatmap of `a(i)`.
- `authority_spatial_montage.png`: downsampled spatial authority maps.
- `authority_step_mean_curve.png`: mean contextual authority over denoising.
- `authority_summary.csv`: per-layer summary statistics.
