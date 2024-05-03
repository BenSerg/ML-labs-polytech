import numpy as np
from distr_abc import Distribution


class NormalDistribution(Distribution):
    def __init__(self, mu=0, sigma=1, size=1) -> None:
        if sigma < 0 or size < 1:
            raise ValueError
        self.mu, self.sigma = mu, sigma
        u1, u2 = np.random.random(size), np.random.random(size)
        self.val = mu + sigma * np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

    def mean_T(self) -> float:
        return self.mu

    def mean(self) -> float:
        return np.mean(self.val)

    def var_T(self) -> float:
        return self.sigma ** 2

    def var(self) -> float:
        return np.var(self.val)
