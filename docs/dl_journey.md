# DL agent training journey

How the PPO agent went from −52 (random policy) to a mean of **855.8**
across seeds 0–9, and what we learned about frame stacking.

Final frozen config:
- **Algorithm**: PPO (Proximal Policy Optimization) via Stable-Baselines3 2.8.0
- **Policy**: CnnPolicy (NatureCNN — 3 conv layers + MLP head)
- **Frame stacking**: last 4 frames concatenated along the channel axis
  (3 → 12 input channels), via `VecFrameStack(n_stack=4)`
- **Training**: 2M steps, 8 parallel envs, Kaggle T4 GPU, ~6 hours
- **Hyperparams**: lr=1e-4, n_steps=512, batch_size=128, n_epochs=10,
  gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0

## The journey

| # | Change | Mean | Good (≥700) | Full laps (≥900) | Notes |
|---|---|---:|---:|---:|---|
| 0 | Random policy (0 steps) | −52 | 0 | 0 | drives in circles |
| 1 | 500k steps, no frame stack, local CPU (killed early) | ~0 | 0 | 0 | too slow locally, moved to Kaggle |
| 2 | 500k steps, no frame stack, Kaggle T4 | **197** | 0 | 0 | learned something but heavily undertrained |
| 3 | 2M steps, no frame stack, Kaggle T4 | **402** | 1 | 0 | doubled, but plateaued — no velocity info |
| 4 | 2M steps, **with frame stack (4 frames)**, Kaggle T4 | **855.8** | 9 | 3 | frame stacking was the key |

## The frame stacking lesson

### The problem without frame stacking

A single 96×96 RGB frame is a photograph — it shows *where* the road is
but not *how fast* the car is going or *which direction* it's currently
turning. The classical agent doesn't need velocity because we hand-coded
the throttle schedule (gas = 0.35 on straights, 0.1 in turns). But the
CNN has to figure out the gas/brake policy from scratch, and from a still
image there's no signal for "I'm going too fast for this turn."

Result: the no-frame-stack agent learns to steer somewhat (it stays on
track more than random) but can't learn to brake. It enters every turn at
full speed, oversteers, clips the grass, and recovers slowly. Mean ~400.

### The fix

Stack the last 4 frames as one input: 4 × 96 × 3 = 12 channels. Now the
CNN can compare frame[t] to frame[t−3] and see:
- Track tiles got bigger → car is moving forward fast
- The road shifted left between frames → car is turning right
- The car sprite barely moved → car is slow (just braked)

This is exactly what "optical flow" computes explicitly in classical CV,
but here the CNN learns to extract it implicitly from the raw pixel
differences. No hand-coded flow computation needed.

Result: mean jumps from 402 to 855, 3 full laps, std drops. The CNN
learned both steering AND speed control from the frame differences.

### Why we didn't figure this out from the start

Frame stacking is standard for Atari/CarRacing in the SB3 RL Zoo, but
the default `CnnPolicy` doesn't include it — you have to add the
`VecFrameStack` wrapper yourself. The SB3 docs mention it but don't
flag it as "you will get garbage without this." We lost two training
runs (~8 hours of Kaggle GPU) learning this the hard way.

## Head-to-head: classical vs DL (final)

| Seed | Classical | DL | Δ |
|---:|---:|---:|---:|
| 0 | 784.0 | 930.8 | +147 |
| 1 | 874.5 | 804.0 | −70 |
| 2 | 470.1 | 893.0 | +423 |
| 3 | 892.6 | 893.4 | +1 |
| 4 | 867.3 | 871.7 | +4 |
| 5 | 741.9 | 896.3 | +154 |
| 6 | 773.2 | 928.1 | +155 |
| 7 | 802.8 | 935.7 | +133 |
| 8 | 808.4 | 830.5 | +22 |
| 9 | 847.4 | 574.4 | −273 |

| Metric | Classical | DL |
|---|---:|---:|
| **Mean** | **786.2** | **855.8** |
| Std | 115.0 | 102.2 |
| Good (≥700) | 9 | 9 |
| Full laps (≥900) | 0 | 3 |
| ms/frame | 4.68 | 5.06 |

### Where DL wins

- **Seed 2** (+423): the T-junction where classical's centerline averaging
  breaks. The CNN doesn't average columns — it learned track shape directly
  from pixels, so the crossing doesn't confuse it.
- **Seeds 0, 6, 7** (full laps, >900): the CNN learned smooth braking into
  turns — something our P-only classical controller can't do.
- **Seeds 5, 6** (~+155 each): tracks with sharp turns that needed speed
  awareness the classical agent lacks.

### Where classical wins

- **Seed 9** (−273): DL's worst track. The policy falls off the track on
  this specific layout. Likely an undertrained region of the policy —
  this track geometry is underrepresented in the 2M training rollouts.
- **Seed 1** (−70): small loss. Classical's 874 is strong here; DL's 804
  is still good but slightly worse.

### The tradeoffs

| | Classical | DL |
|---|---|---|
| **Training time** | 0 (hand-tuned) | ~6 hours on T4 GPU |
| **Engineering time** | ~4 hours of tuning HSV, gains, fixing bugs | ~20 min of code + waiting |
| **Interpretability** | Full — every decision is a named constant | None — it's a black box |
| **Robustness to unseen tracks** | Depends on HSV range | Depends on training distribution |
| **Velocity awareness** | None (hand-coded throttle schedule) | Yes (learned from frame diffs) |
| **Failure mode** | Predictable: stalls, grass drive, T-junctions | Unpredictable: just stops working on some seeds |

## Training infrastructure notes

- **Local Mac CPU**: ~58 fps with 8 envs. 500k steps ≈ 2.5 hours. Too slow
  for iteration.
- **Kaggle T4 GPU**: ~333k steps/hour. 500k ≈ 1.5 hours. 2M ≈ 6 hours.
  Well within the 12-hour session limit.
- **SB3 auto-selects CPU on Mac** — it intentionally skips Apple Silicon's
  MPS backend because of known PyTorch correctness bugs on RL workloads.
- Checkpoint size: ~28 MB (`.zip` containing policy weights + optimizer
  state + normalization stats).

## Open questions for the robustness sweep

1. **Gaussian pixel noise**: classical's HSV thresholding will degrade
   because noise pushes gray pixels across the saturation boundary.
   DL's CNN may be more robust (conv layers are natural low-pass filters)
   or equally fragile. Unknown.

2. **Hue shift**: classical WILL break — the HSV thresholds are hard-coded
   for CarRacing's default gray/yellow palette. Shift the hue by 30° and
   the mask will be empty. DL might generalize if the training data had
   enough color variation, or might fail equally hard. Unknown.

Both worth testing. These are the results that will make the eval section
of the video interesting.
