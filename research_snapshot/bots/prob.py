"""Shared probability utilities for Bluff bots."""

import math


class Hypergeometric:
    """Exact hypergeometric probability calculations.

    P(X=k) where X ~ Hypergeometric(N, K, n).
    """

    @staticmethod
    def pmf(k: int, N: int, K: int, n: int) -> float:
        if k > K or k > n or K > N or n > N or k < 0:
            return 0.0
        return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)

    @staticmethod
    def cdf(k: int, N: int, K: int, n: int) -> float:
        return sum(Hypergeometric.pmf(i, N, K, n) for i in range(min(k, K) + 1))
