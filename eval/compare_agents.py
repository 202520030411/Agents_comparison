"""Head-to-head: classical vs DL agent on the same 10 seeds.

Both agents see the same CarRacing-v3 env with the same seed, reset the
same way. Measures:
- per-episode return
- mean, std, good-seed count (>=700), full-lap count (>=900)
- wall-clock ms per frame (compute cost)
"""
import time
import numpy as np

import classical.controller as C
from classical.run import run as run_classical
from dl.infer import run as run_dl


SEEDS = list(range(10))


def _timed(fn, seed):
    t0 = time.time()
    ret = fn(seed, False)
    elapsed = time.time() - t0
    # CarRacing-v3 episodes run up to 1000 frames (most of ours do).
    # We approximate ms/frame by assuming 1000 frames; close enough for
    # a compute-cost comparison between agents.
    return ret, elapsed * 1000.0 / 1000.0  # ms per frame


def main():
    # Make sure classical uses its frozen best config.
    C.K_P_HEADING = 3.4

    rows = []
    for seed in SEEDS:
        cls_ret, cls_ms = _timed(run_classical, seed)
        dl_ret, dl_ms = _timed(run_dl, seed)
        rows.append((seed, cls_ret, cls_ms, dl_ret, dl_ms))
        print(f"seed={seed:2d}  classical={cls_ret:7.1f} ({cls_ms:5.2f} ms/f)  "
              f"dl={dl_ret:7.1f} ({dl_ms:5.2f} ms/f)")

    cls_rets = np.array([r[1] for r in rows])
    dl_rets = np.array([r[3] for r in rows])
    cls_ms = np.array([r[2] for r in rows])
    dl_ms = np.array([r[4] for r in rows])

    print()
    print(f"{'metric':<20}{'classical':>12}{'dl':>12}")
    print("-" * 44)
    print(f"{'mean return':<20}{cls_rets.mean():>12.1f}{dl_rets.mean():>12.1f}")
    print(f"{'std return':<20}{cls_rets.std():>12.1f}{dl_rets.std():>12.1f}")
    print(f"{'good (>=700)':<20}{(cls_rets >= 700).sum():>12d}{(dl_rets >= 700).sum():>12d}")
    print(f"{'full lap (>=900)':<20}{(cls_rets >= 900).sum():>12d}{(dl_rets >= 900).sum():>12d}")
    print(f"{'ms / frame':<20}{cls_ms.mean():>12.2f}{dl_ms.mean():>12.2f}")


if __name__ == "__main__":
    main()
