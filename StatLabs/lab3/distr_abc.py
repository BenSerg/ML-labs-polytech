from abc import ABC


class Distribution(ABC):
    def mean(self) -> float:
        raise NotImplementedError

    def var(self) -> float:
        raise NotImplementedError
