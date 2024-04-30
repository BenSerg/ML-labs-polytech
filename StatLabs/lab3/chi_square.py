import numpy as np
from distr_abc import Distribution
from normal import NormalDistribution


class ChiSquaredDistribution(Distribution):
    def __init__(self, N, size=1) -> None:
        self.N = N
        self.val = np.array([np.sum(NormalDistribution(mu=0, sigma=1, size=N).val ** 2) for _ in range(size)])

    def mean_T(self) -> float:
        return self.N

    def var_T(self) -> float:
        return 2 * self.N

    def mean(self) -> float:
        return np.mean(self.val)

    def var(self) -> float:
        return np.var(self.val)
