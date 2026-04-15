"""Planner: pick a point up the road to aim at, and return the angle to it.

The car sprite is always drawn at the same spot in the picture (row 70, col 48),
so we don't need to track where the car "is" — only where we want it to go next.

Adaptive lookahead: instead of a fixed "aim 35 pixels ahead," we look at how
spread out the middle-of-road columns are (straight road = all similar; curvy
road = big spread), then aim far on straights and close in curves.
"""
import numpy as np

CAR_ROW = 70   # the car sprite sits here in the 96x96 frame
CAR_COL = 48

LOOKAHEAD_MAX = 55   # on a dead-straight road, aim this far up the picture
LOOKAHEAD_MIN = 15   # in the tightest turns, aim this close to the car
SMOOTH = 0.8         # how much of the previous curvy-number to keep each frame

# Sanity check: the target point must be AHEAD of the car, not below it.
# We never try to aim at a row greater than CAR_ROW (those rows are at/under
# the car sprite, they describe where the car already is, not where it's going).
# If the topmost visible road row is still at/below the car (seed 2 grass case),
# there's no forward target at all -- return None and let fallback take over.
MIN_AHEAD_ROWS = 5   # topmost visible row must be at least 5 rows ABOVE the car


class Planner:
    """Stateful planner so we can smooth the curvy-number across frames."""

    def __init__(self):
        self.smoothed_curvy = 0.0

    def target(self, centerline_pts: list[tuple[int, int]]) -> tuple[float, float, int] | None:
        """Return (lateral_error_px, heading_err_rad, lookahead_used) or None."""
        if not centerline_pts:
            return None

        # 1. Measure how curvy the road looks right now: spread of the middle-of-road
        #    columns across all visible rows. Straight road -> small number. Curvy -> big.
        cols = np.fromiter((c for _, c in centerline_pts), dtype=np.float32)
        curvy = float(cols.std())

        # 2. Smooth it so a noisy frame doesn't flip the lookahead. Each frame we keep
        #    80% of the old number and let the new measurement nudge it 20%.
        self.smoothed_curvy = SMOOTH * self.smoothed_curvy + (1 - SMOOTH) * curvy

        # 3. Turn the smoothed curvy number into a lookahead: far on straights, close
        #    in curves. Linear slide between LOOKAHEAD_MAX and LOOKAHEAD_MIN.
        lookahead = int(np.clip(LOOKAHEAD_MAX - self.smoothed_curvy, LOOKAHEAD_MIN, LOOKAHEAD_MAX))

        # 4. Find a centerline point to aim at.
        #    Ideal: the point closest to (car_row - lookahead).
        #    But only consider points that are actually AHEAD of the car.
        ahead_pts = [rc for rc in centerline_pts if rc[0] < CAR_ROW - MIN_AHEAD_ROWS]
        if not ahead_pts:
            # No visible road ahead of us -- seed 2 style "we're in grass with
            # only the trailing edge of where we came from still visible".
            return None

        target_row = CAR_ROW - lookahead
        pt = min(ahead_pts, key=lambda rc: abs(rc[0] - target_row))
        r, c = pt

        # 5. Angle from the car to that point.
        dx = c - CAR_COL                              # + means target is right of car
        dy = CAR_ROW - r                              # + means target is ahead (up)
        heading_err = float(np.arctan2(dx, max(dy, 1)))
        return float(dx), heading_err, lookahead
