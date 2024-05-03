import numpy as np
from distr_abc import Distribution
from normal import NormalDistribution
from chi_square import ChiSquaredDistribution


class StudentDistribution(Distribution):
    def __init__(self, N=1, size=1) -> None:
        self.N = N
        z = NormalDistribution(mu=0, sigma=1, size=size).val
        chi = ChiSquaredDistribution(N, size=size).val
        self.val = z / np.sqrt(chi / N)

    def var_T(self) -> float:
        return self.N / (self.N - 2)

    def mean_T(self) -> float:
        return 0

    def mean(self) -> float:
        return np.mean(self.val)

    def var(self) -> float:
        return np.var(self.val)
