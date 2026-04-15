"""PID-style controller: maps (lateral_error, heading_error) -> (steer, gas, brake)."""
import numpy as np

K_P_HEADING = 2.2
K_P_LATERAL = 0.03
MAX_GAS = 0.35
BRAKE_CURVATURE = 0.6   # |heading_err| above this triggers brake


class Controller:
    def __init__(self):
        self.last_steer = 0.0

    def step(self, lateral_err: float, heading_err: float) -> np.ndarray:
        steer = K_P_HEADING * heading_err + K_P_LATERAL * lateral_err
        steer = float(np.clip(steer, -1.0, 1.0))

        if abs(heading_err) > BRAKE_CURVATURE:
            # Leave a little gas on so the car never stops rolling — a fully
            # stopped car on a curve is a death trap (same frame forever).
            gas, brake = 0.1, 0.1
        else:
            gas = MAX_GAS * (1.0 - abs(heading_err) / BRAKE_CURVATURE)
            brake = 0.0

        self.last_steer = steer
        return np.array([steer, gas, brake], dtype=np.float32)

    def fallback(self) -> np.ndarray:
        """No track visible: steer in the last known direction and coast gently forward."""
        recover = float(np.clip(self.last_steer * 1.5, -1.0, 1.0))
        return np.array([recover, 0.15, 0.0], dtype=np.float32)
