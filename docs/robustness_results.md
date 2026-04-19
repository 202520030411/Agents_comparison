# Robustness sweep results

Both agents evaluated under two types of image perturbation, on seeds
{0, 3, 5, 7, 9}. The perturbation is applied to every frame the agent
sees — the underlying physics and rewards are unchanged.

## Gaussian pixel noise

Add random noise drawn from N(0, σ) to every pixel, then clip to [0, 255].
Models camera sensor noise, low light, or video compression artifacts.

| Noise σ | Classical | DL | Winner |
|---:|---:|---:|---|
| 0 (clean) | 813.8 | 846.1 | DL |
| 10 (mild) | 774.4 | 814.0 | DL |
| 25 (moderate) | 826.5 | 878.7 | DL |
| 50 (heavy) | **−44.5** | **867.2** | **DL** |

**DL wins across all noise levels. Classical collapses at σ=50.**

### Why classical breaks

The HSV mask uses a hard boundary: saturation ≤ 60 means "gray, probably
road." When we add noise with σ=50, some road pixels get bumped to
saturation 65 and are rejected (false negative), while some grass pixels
drop to saturation 55 and are accepted (false positive). The mask
flickers randomly every frame — the centerline jumps around, the planner
aims at garbage, and the car drives off.

At σ=10 and σ=25 the noise isn't large enough to push many pixels across
the threshold, so classical holds steady. At σ=50 it crosses the tipping
point and collapses to −44 (worse than doing nothing).

### Why DL survives

The CNN's convolutional layers naturally average over small local regions
(each filter covers a 3×3 or 8×8 patch). Random pixel noise mostly
cancels out within each patch, so the learned features remain stable.
The frame stacking (4 frames averaged implicitly) further smooths the
noise. The DL agent scores 867 at σ=50 — essentially unchanged from
clean (846).

## Hue shift

Rotate the hue channel of every pixel by a fixed amount (0–179 in OpenCV
scale). Models a recolored track — as if the game used a different
palette (desert, snow, night).

| Hue shift | Classical | DL | Winner |
|---:|---:|---:|---|
| 0 (default) | 813.8 | 846.1 | DL |
| 15 | **775.8** | **−82.8** | **Classical** |
| 30 | **743.3** | **−93.1** | **Classical** |
| 60 | **817.7** | **−93.1** | **Classical** |

**Classical wins across all nonzero hue shifts. DL collapses immediately.**

### Why DL breaks

The CNN memorized the exact pixel colors from its 2M training frames.
It has never seen a blue road or a purple stripe — those pixel patterns
are completely outside its training distribution. At hue shift 15 the
road looks subtly different; by shift 30 it looks radically different.
The policy outputs random-looking actions and the car drives off
immediately (−93 ≈ doing nothing for the full episode).

This is the standard deep learning brittleness to distribution shift:
the model works perfectly on data that looks like training, and
catastrophically on data that doesn't.

### Why classical survives

The HSV thresholds check saturation (≤ 60 for gray) and value (40–200
for brightness), but NOT hue. Rotating the hue from 0 to 60 changes
which color the road "is" (gray → blue-gray → teal) but doesn't change
its saturation or brightness. The mask still fires correctly.

The one exception would be a hue shift that turns the road into the
same hue as the grass (green, hue ≈ 60 in OpenCV). At shift=60 we're
near that territory, but classical still scores 817 because the grass
has HIGH saturation (vivid green) while the road has LOW saturation
(grayish teal), so the saturation check still separates them.

## The takeaway

```
                    Noise robust?     Color robust?
Classical               ✗                 ✓
DL                      ✓                 ✗
```

Neither agent is strictly better. They have complementary failure modes:

- **Classical** fails when pixel values are noisy, because thresholding
  is a hard boundary and noise pushes pixels across it randomly.
- **DL** fails when colors change, because it memorized the training
  palette and has zero generalization to unseen color schemes.

A truly robust agent would need either:
1. **Data augmentation during DL training** — train with random hue
   shifts so the CNN learns color-invariant features.
2. **A hybrid architecture** — use classical saturation-based detection
   (color-robust) with learned smoothing (noise-robust).
3. **Domain randomization** — randomize the environment's visuals during
   training so the policy can't memorize any single appearance.

## Evaluation details

- Seeds evaluated: {0, 3, 5, 7, 9} (5 seeds, subset of the full 10)
- Each number is the mean return across those 5 seeds
- Classical config: adaptive lookahead, gain 3.4, stall-trap fix
- DL config: PPO CnnPolicy, 4-frame stack, 2M steps, Kaggle T4
- Perturbation wrappers: env/perturbations.py (NoisyObsWrapper, HueShiftWrapper)
- Sweep script: eval/robustness_sweep.py
