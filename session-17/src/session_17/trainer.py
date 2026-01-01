from abc import ABC
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrainerStats:
    step: int
    episode: int
    reward: float
    others: dict[str, Any] = field(default_factory=dict)


class Trainer(ABC):
    def step(self) -> tuple[TrainerStats, NDArray[np.uint8] | None]:
        raise NotImplementedError
