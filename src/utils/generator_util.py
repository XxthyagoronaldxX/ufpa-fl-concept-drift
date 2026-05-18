import numpy as np

from config import SEED


class GeneratorUtil:
    """Utilitário centralizado para geração de números aleatórios."""

    _rng = np.random.default_rng(seed=SEED)

    @staticmethod
    def reseed(seed=SEED):
        """Reinicializa o gerador global (opcional para experimentos)."""
        GeneratorUtil._rng = np.random.default_rng(seed=seed)

    @staticmethod
    def exponential(scale, size=None, rng=None):
        if rng is None:
            rng = GeneratorUtil._rng
        return rng.exponential(scale, size)

    @staticmethod
    def poisson(lam, size=None, rng=None):
        if rng is None:
            rng = GeneratorUtil._rng
        return rng.poisson(lam, size)

    @staticmethod
    def normal(loc=0.0, scale=1.0, size=None, rng=None):
        if rng is None:
            rng = GeneratorUtil._rng
        return rng.normal(loc, scale, size)

    @staticmethod
    def binomial(n, p, size=None, rng=None):
        if rng is None:
            rng = GeneratorUtil._rng
        return rng.binomial(n, p, size)

    @staticmethod
    def permutation(n, rng=None):
        if rng is None:
            rng = GeneratorUtil._rng
        return rng.permutation(n)
