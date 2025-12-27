from abc import ABC
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class TrainerStats:
    step: int
    episode: int
    reward: float
    temperature: float


class Trainer(ABC):
    def step(self) -> tuple[TrainerStats, NDArray[np.uint8] | None]:
        raise NotImplementedError
