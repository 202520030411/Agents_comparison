# Classical agent tuning journey

How the hand-coded CarRacing agent went from 141 (seed 0 only, broken) to a
mean of **786.2** across seeds 0–9, and what we learned at each step.

Final frozen config:
- **Perception**: HSV threshold on gray asphalt OR yellow centerline, 3×3
  morph-close, row-wise centroid of white pixels.
- **Planner**: adaptive lookahead in [15, 55] rows, computed from the std of
  the centerline column values (straight → big lookahead, curvy → small).
  Smoothed frame-to-frame with `new = 0.8*old + 0.2*measured`. Sanity check:
  only aim at centerline points that are at least 5 rows *above* the car.
- **Controller**: P on heading angle (gain 3.4) + tiny P on lateral error
  (gain 0.03). Throttle slides from 0.35 on straights to 0.1 in sharp turns;
  brake = 0.1 in sharp turns (never 0 — see the stall-trap bug below).
- **Fallback**: when planner returns None, keep steering in the last known
  direction ×1.5 and coast at gas=0.15.

## The journey, step by step

| # | Change | Mean | Good (≥700) | Std | Notes |
|---|---|---:|---:|---:|---|
| 0 | Initial: narrow HSV box, lookahead 20, gain 1.2, brake-only in turns | 141¹ | — | — | seed 0 only |
| 1 | Widen HSV to include bright asphalt + yellow stripe | 144 | — | — | perception fixed, controller still bad |
| 2 | Lookahead 20 → 35, gain 1.2 → 2.2, max gas 0.6 → 0.35 | 777¹ | — | — | seed 0 only, huge jump |
| 3 | Sweep seeds 0–9 on step 2 config | **502** | 4 | 310 | first real baseline |
| 4 | + adaptive lookahead (15–55) with gain 2.2 | 496 | 4 | — | looked like a wash (it wasn't — see step 6) |
| 5 | + retune gain to 3.4 | 602 | 6 | — | real gain from adaptive shows up |
| 6 | + stall-trap fix: gas=0.1 in brake branch (was 0.0) | **751** | 9 | 215 | whole curve shifted up, seed 5 and seed 6 recovered |
| 7 | + "aim only at points ahead of car" sanity check (seed 2 fix) | **786** | 9 | **115** | seed 2 went 120 → 470, std halved |

¹ seed 0 only; not a 10-seed mean.

## Three bugs worth remembering

### Bug A — HSV window missed the asphalt

**Symptom**: seed 0 return stuck at ~141. Mask looked almost empty on the
diagnostic dumps even though the frame clearly showed lots of gray road.
Also: on straights, the mask had a *vertical gap down the middle* because
the yellow centerline stripe was excluded by the "low saturation" filter.

**Root cause**: initial thresholds were `V` in [80, 140], but CarRacing's
asphalt is brighter than 140. And the yellow stripe isn't gray at all — it's
hue ~25 with high saturation, so "low saturation" rejects it.

**Fix**: widen gray to V in [40, 200], and OR with a separate yellow range
[hue 15–40, sat ≥ 80, val ≥ 80]. Mean went 141 → 144 on seed 0 alone —
*but perception was now trustworthy*, which unlocked every later fix.

**Lesson**: debug masks visually before touching any downstream component.
Score changes don't tell you whether perception is right.

### Bug B — the stall trap (seeds 5 and 6)

**Symptom**: seed 5 = 207 with the adaptive+gain-3.4 config, worse than the
fixed-lookahead version's 738. Seed 6 was −82 across *every* config I tried.

**Diagnosis**: instrumented a per-frame dump. From t≈400 onward every row
was **pixel-identical** — `n_pts=79  dx=25  hdg=0.62  steer=1.00  gas=0.00
brake=0.30`. For 600 frames. The car had stopped.

Why: the brake branch was `gas=0, brake=0.3`. The car decelerated into a
sharp turn until velocity reached zero. Once stopped, perception sees the
same frame forever → same angle → same "brake" decision → car stays stopped
forever. The episode just bleeds −0.1/frame until timeout.

**Fix**: one-character change, `gas=0.1, brake=0.1`. The car can never fully
stop, so it always rolls through turns.

**Lesson**: a downstream bug (in the controller) was making an upstream
improvement (adaptive lookahead) look worthless. At step 4 I thought
adaptive was ~equal to fixed; in reality *both* were being capped by the
stall trap, just on different seeds. The "it's a wash" conclusion was a
lie. If a change that should help comes back roughly equal, check whether
something downstream is swallowing the difference.

### Bug C — drives into grass because perception "succeeded" (seed 2)

**Symptom**: seed 2 stuck at ~120 across every config. The car would drive
fine for ~300 frames, then confidently accelerate in a straight line into
the grass and get terminated at t≈484.

**Diagnosis**: around t=300 the car passed through what looks like a
T-junction where the track visits itself (two parallel segments visible in
the same frame). The row-wise centroid averages *across both segments*, so
"middle of the road" lands in the grass between them. Car aims there and
drives off.

Once off, at t=340 the mask has 12 white pixels, all clustered near the
bottom of the image (the trailing edge of the track the car just left). The
planner calls `min(pts, key=lambda rc: abs(rc[0] - 20))` — there are no
points anywhere near row 20, but `min` returns *something*: the row-75
point. Its column is ~48 (below the car), so `dx=0, hdg=0`. Controller
says: "perfectly aimed, full gas ahead." Ahead is more grass.

**Fix**: only consider centerline points that are actually *above* the car
(`row < CAR_ROW - 5`). If nothing remains, return None and let fallback
take over. Seed 2: 120 → 470. No other seed regressed.

**Lesson**: `min(...)` over an arbitrary list is dangerous when the list
might not contain the thing you're looking for. You need an explicit check
that the result actually satisfies your requirement, not just "is closest
to satisfying it."

A false alarm along the way: I first tried `MIN_POINTS < 25` and
`MAX_TARGET_ROW_MISS > 10` as the check. Both were too blunt — they fired
on sharp turns where the driver only saw rows 50–84 of road, rejecting
legitimate frames. Mean dropped 751 → 622. Fixing the check to be about
"forward-of-car" instead of "close-to-target-row" was the difference.

## Seed-by-seed comparison, before vs. final

| Seed | Step 3 (fixed lookahead 35, gain 2.2) | Step 7 (final) | Δ |
|---:|---:|---:|---:|
| 0 | 846.7 | 784.0 | −63 |
| 1 | 685.5 | 874.5 | +189 |
| 2 | 156.7 | 470.1 | **+313** |
| 3 | 497.8 | 892.6 | **+395** |
| 4 | 641.8 | 867.3 | +225 |
| 5 | 154.5 | 741.9 | **+587** |
| 6 | −2.8 | 773.2 | **+776** |
| 7 | 809.1 | 802.8 | −6 |
| 8 | 828.3 | 808.4 | −20 |
| 9 | 840.4 | 847.4 | +7 |
| **mean** | **545.8** | **786.2** | **+240** |
| good (≥700) | 4 | 9 | +5 |
| std | 310 | 115 | −195 |

Five seeds moved by more than 200 points. All in the right direction.

## Open issues we deliberately didn't fix

- **No seed clears 900** (a true full lap). The 800s are "finished the lap
  but clipped a few tiles." Getting to 900+ would need a D-term to damp
  steering oscillation on straights, and probably speed awareness in the
  controller.
- **Seed 2 still bottoms out at 470**, not 700+. The sanity check stops the
  crash but the fallback is a dumb "keep drifting in last direction" —
  it doesn't actively search for the lost track. Fixing it would need
  real recovery logic.
- **`K_P_LATERAL = 0.03`** was never touched after the first tuning pass
  and is probably off for the adaptive-lookahead regime.

These are all under ~100 points of expected improvement each, and the
real signal in this project will come from comparing classical vs. DL,
not from polishing classical from 786 to 850.
