"""Per-episode metrics shared by both agents."""
from dataclasses import dataclass


@dataclass
class EpisodeStats:
    seed: int
    agent: str
    total_return: float
    frames: int
    off_track_frames: int
    wall_ms_per_frame: float

    @property
    def success(self) -> bool:
        return self.total_return > 900.0
