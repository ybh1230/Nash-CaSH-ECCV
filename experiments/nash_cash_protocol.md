# Nash-CaSH Experiment Protocol

This protocol is for the new Nash-CaSH submission. Do not claim improvement until all rows are evaluated with fixed prompts, fixed seeds, official metrics, and matched runtime settings.

## Main 1080P Setting

- Backbone: Wan2.1-1.3B-Diffusers.
- Resolution: 1920 x 1088.
- Prompts: 60 randomly selected VBench prompts.
- Seeds: 5 seeds per prompt.
- Inference steps: 50.
- Strength: 0.7.
- Guidance scale: 5.0.
- Flow shift: 9.0.
- Metrics: VBench subject consistency, background consistency, motion smoothness, aesthetic quality, imaging quality, overall consistency, and overall score.

## Rows To Run

- Native high-resolution inference.
- Local-window only.
- Global full-attention only when feasible.
- Fixed local-global override: `--no-nash_cash`.
- Nash-CaSH without cache: `--mode nocache --nash_cash`.
- Nash-CaSH with cache: `--mode cache --cache_steps 2 --nash_cash`.
- Nash-CaSH cache sweep: `--cache_steps 5` and `--cache_steps 8`.

## Required Ablations

- Nash disabled versus enabled.
- Nash floor: 0.55, 0.65, 0.75.
- Nash ceiling: 0.90, 0.98, 1.00.
- Global prior: 1.0, 1.25, 1.5.
- Local prior: 0.75, 1.0, 1.25.
- Cache interval: 2, 5, 8.

## Reporting Rules

- Report mean and standard error over seeds.
- Keep prompts, seeds, model versions, PyTorch version, Diffusers version, GPU model, and offload policy in the appendix.
- Measure runtime on the same GPU with the same precision.
- Fill paper tables only with measured values from this protocol.
