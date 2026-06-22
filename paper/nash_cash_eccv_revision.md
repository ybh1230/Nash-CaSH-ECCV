# Nash-CaSH: Game-Theoretic Cross-Attention Harmonization for Training-Free Ultra-High-Resolution Video Generation

## Abstract

Generating ultra-high-resolution videos with pretrained video Diffusion Transformers is difficult because attention layers must operate far beyond the token scale observed during pretraining. Local window attention improves detail fidelity by preserving a native-scale receptive field, yet it can weaken global semantic coordination and create repeated structures. We propose **Nash-CaSH**, a training-free cross-attention harmonization strategy that treats local and global semantic evidence as two cooperative players in a Nash bargaining game. At each cross-attention layer, Nash-CaSH computes branch payoffs from query-output compatibility and response confidence, then derives a closed-form Nash social-welfare allocation to fuse the global branch and local branch. This adaptive equilibrium gate replaces fixed hand-tuned branch mixing and lets each token receive as much global semantic authority as it needs while retaining local detail when it is beneficial. Nash-CaSH is compatible with cached global cross-attention, enabling efficient high-resolution inference without model finetuning. We provide an implementation and an evaluation protocol for 1080P and 3K video generation using official VBench metrics.

## 1. Introduction

Video Diffusion Transformers have become a strong backbone for text-to-video generation, but their attention cost grows rapidly with spatial and temporal resolution. A model pretrained at native resolution can often generate plausible low-resolution videos, yet direct high-resolution inference introduces a distribution shift in token count and receptive field. The result is usually a trade-off: full attention preserves long-range structure but becomes expensive and may blur local details, while local attention improves texture and sharpness but can create repeated objects or inconsistent global layouts.

This work studies a training-free solution for high-resolution video synthesis. The central idea is to preserve local native-scale evidence while still injecting global semantic coordination. Instead of using a fixed mixing coefficient between the local and global branches, we formulate the interaction as a token-wise bargaining problem. The local branch is responsible for training-scale detail fidelity; the global branch is responsible for semantic layout. Their relative influence should vary across denoising steps, layers, prompts, and spatial positions.

We introduce Nash-CaSH, a Nash-equilibrium cross-attention harmonizer. For every cross-attention output, Nash-CaSH estimates the utility of each branch and solves a Nash social-welfare allocation. The resulting gate is adaptive, bounded, and closed-form, so it adds no trainable parameters and requires no optimization loop. The method also preserves compatibility with feature caching: a cached global candidate can be reused across steps, while the current local candidate still participates in the equilibrium allocation.

Our contributions are:

- We formulate cross-attention branch fusion for high-resolution video generation as a Nash bargaining problem.
- We propose a closed-form training-free gate that computes token-wise equilibrium weights from branch payoffs.
- We integrate the gate into both non-cached and cached high-resolution inference paths.
- We provide complete code, ablation scripts, and an ECCV-style evaluation protocol for reproducible measurement.

## 2. Related Work

### Training-Free High-Resolution Generation

Training-free high-resolution generation aims to extend pretrained diffusion models to larger output sizes without finetuning. Existing strategies include coarse-to-fine refinement, resolution-aware attention scaling, positional adjustment, and local receptive-field control. These methods reduce training cost but often struggle when the target resolution greatly exceeds the native training scale, especially for videos where temporal dependencies multiply the attention burden.

### Local and Global Attention Trade-Offs

Local attention reduces computation and preserves detail by limiting each query to a native-scale neighborhood. However, purely local interactions can weaken long-range structure. Global attention provides holistic context but is expensive at high resolution and can dilute fine-grained visual evidence. Nash-CaSH is designed for this local-global trade-off: it allows both branches to participate in the final semantic decision instead of assigning a fixed winner.

### Game-Theoretic Allocation

Nash bargaining and Nash social welfare provide principled tools for allocating a shared resource between players with positive utilities. In Nash-CaSH, the shared resource is semantic authority in cross-attention. The resulting allocation is fair in the sense that each branch's influence grows with its measured payoff for the current token.

## 3. Method

### 3.1. Coarse-to-Fine High-Resolution Inference

Given a prompt, the model first generates a base video at the native resolution of the pretrained backbone. The video is then upsampled in pixel space, encoded into latents, and perturbed with Gaussian noise following an SDEdit-style refinement step. High-resolution denoising begins from this noisy latent, preserving the base layout while allowing high-frequency details to be regenerated.

### 3.2. Local-Global Dual Branches

During high-resolution denoising, the latent is duplicated into two branches. The local branch uses inward sliding-window self-attention, keeping the receptive field of each query close to the native training scale. The global branch provides holistic context using full attention. This design gives the model access to both local detail and global semantic structure.

Let the cross-attention outputs before output projection be

```math
O_l(q), O_g(q) \in R^{H \times D},
```

where `q` indexes a token, `l` denotes the local branch, `g` denotes the global branch, `H` is the number of heads, and `D` is the head dimension.

### 3.3. Branch Payoffs

Nash-CaSH measures the utility of each branch using two training-free signals: query compatibility and normalized response confidence. For branch `b in {l,g}`, with query `Q_b(q)` and cross-attention output `O_b(q)`, the payoff is

```math
u_b(q) =
\pi_b
\left[
\frac{1 + cos(vec(O_b(q)), vec(Q_b(q)))}{2}
+
\sigma\left(
\frac{rms(O_b(q))}{mean_q rms(O_b(q)) + \epsilon} - 1
\right)
+
\epsilon
\right].
```

The cosine term rewards outputs aligned with the current query. The response-confidence term rewards branch outputs that are strong relative to the layer's average response. The prior `pi_b` can encode a conservative preference for global semantic stability. The default values are `pi_g = 1.25` and `pi_l = 1.0`.

### 3.4. Nash Equilibrium Gate

Let `a(q)` be the semantic authority assigned to the global branch for token `q`; the local branch receives `1-a(q)`. Nash-CaSH solves

```math
a^*(q)
= argmax_{a in [0,1]}
u_g(q) log(a + epsilon)
+ u_l(q) log(1-a + epsilon).
```

This objective has the closed-form ratio

```math
r(q)
=
\frac{u_g(q)^{1/tau}}
{u_g(q)^{1/tau} + u_l(q)^{1/tau}},
```

where `tau` is a temperature. To avoid unstable extremes in training-free inference, the ratio is mapped into a bounded interval:

```math
a(q) = a_min + (a_max - a_min) r(q).
```

The fused cross-attention output is

```math
O_{Nash}(q)
= a(q)O_g(q) + (1-a(q))O_l(q).
```

The default interval is `a_min = 0.65`, `a_max = 0.98`. This keeps global semantics active while permitting token-wise local detail recovery.

### 3.5. Cached Nash-CaSH

Full global attention is expensive at high resolution. Nash-CaSH supports cached global cross-attention by refreshing the global candidate every `P` denoising steps. For an intermediate step `t`, the fusion becomes

```math
O_{Nash,t}(q)
= a_t(q)O_{g,t'}(q) + (1-a_t(q))O_{l,t}(q),
```

where `t'` is the most recent refresh step. The local branch is still computed at the current step, so the gate can react to the current latent even when the global candidate is reused.

## 4. Experiments

### 4.1. Evaluation Setup

The final submission should evaluate Nash-CaSH under the following fixed protocol:

- Backbone: Wan2.1-1.3B and Wan2.1-14B.
- Resolution: 1920 x 1088 for 1080P; 3380 x 1920 for 3K.
- Inference steps: 50.
- Strength: 0.7.
- Guidance scale: 5.0.
- Flow shift: 9.0.
- Metrics: VBench subject consistency, background consistency, motion smoothness, aesthetic quality, imaging quality, overall consistency, and overall score.
- Seeds: five seeds per prompt.
- Runtime: measured with the same GPU, precision, offload policy, and software versions.

### 4.2. Main Results Template

The table below is a reporting template. Nash-CaSH numbers must be filled only after official evaluation runs finish.

| Method | Subject | Background | Motion | Aesthetic | Imaging | Overall Consistency | Overall Score | Time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Native high-resolution inference | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Fixed local-global override | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Nash-CaSH w/o cache | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Nash-CaSH w/ cache, P=2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Nash-CaSH w/ cache, P=5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 4.3. Ablation Studies

Required ablations:

- Nash gate disabled versus enabled.
- Fixed override versus Nash-CaSH under the same dual-branch setting.
- `a_min`: 0.55, 0.65, 0.75.
- `a_max`: 0.90, 0.98, 1.00.
- Global prior `pi_g`: 1.0, 1.25, 1.5.
- Cache interval `P`: 2, 5, 8.

### 4.4. Qualitative Analysis

The qualitative section should include prompts with repeated objects, large foreground-background structure, long camera motion, and fine local textures. The expected analysis should compare local-only, global-only, fixed override, and Nash-CaSH outputs. The central claim should be supported by visual examples where Nash-CaSH preserves global layout while recovering sharper local details.

## 5. Conclusion

Nash-CaSH introduces a game-theoretic cross-attention harmonizer for training-free ultra-high-resolution video generation. By converting local-global branch fusion into a Nash social-welfare allocation, the method adaptively balances semantic consistency and local detail without training or extra model parameters. Its cache-compatible design keeps high-resolution inference practical. Final numerical claims should be made only after completing the official evaluation protocol with fixed seeds and reproducible settings.
