import numpy as np
from distr_abc import Distribution


class ExponentialDistribution(Distribution):
    def __init__(self, scale=1, size=1) -> None:
        self.scale = scale
        if scale < 0:
            raise ValueError
        self.val = -scale * np.log(np.random.rand(size))

    def mean_T(self) -> float:
        return self.scale

    def var_T(self) -> float:
        return self.scale ** 2

    def mean(self) -> float:
        return np.mean(self.val)

    def var(self) -> float:
        return np.var(self.val)
