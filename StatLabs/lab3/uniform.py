import numpy as np
from distr_abc import Distribution


class UniformDistribution(Distribution):
    def __init__(self, low=0, high=1, size=1) -> None:
        self.low, self.high = low, high
        if low > high or size < 1:
            raise ValueError
        self.val = (self.high - self.low) * np.random.random(size) + self.low

    def mean_T(self) -> float:
        return (self.low + self.high) / 2.0

    def mean(self) -> float:
        return np.mean(self.val)

    def var_T(self) -> float:
        return (self.high - self.low) ** 2 / 12.0

    def var(self) -> float:
        return np.var(self.val)
