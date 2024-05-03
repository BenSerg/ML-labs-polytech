import math

import numpy as np
from distr_abc import Distribution


class WeibullDistribution(Distribution):
    def __init__(self, scale=1, size=1, m=1) -> None:
        if scale < 0 or size < 1 or m <= 0:
            raise ValueError
        self.scale = scale
        self.m = m
        u = np.random.random(size=size)
        self.val = scale * (-np.log(u)) ** (1 / m)

    def mean_T(self) -> float:
        return self.scale * math.gamma(1 + 1 / self.m)

    def mean(self) -> float:
        return np.mean(self.val)

    def var_T(self) -> float:
        return self.scale ** 2 * (math.gamma(1 + 2 / self.m) - math.gamma(1 + 1 / self.m) ** 2)

    def var(self) -> float:
        return np.var(self.val)

    def cdf(self):
        return self.m * self.scale ** self.m * self.val ** (self.m - 1) * np.exp(-(self.scale * self.val) ** self.m)
