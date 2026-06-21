"""CGSP: CycleGAN Self-Play for two-player zero-sum games.

A pure-JAX research prototype that borrows CycleGAN's cycle-consistency loss as a
regularizer for self-play, aiming to damp the rotational/cycling dynamics that make
naive simultaneous-gradient self-play fail to converge on non-transitive games.
"""

from garip.games import ZeroSumGame, rps, matching_pennies, random_zero_sum
from garip.exploitability import exploitability
from garip import methods, train, kuhn, leduc

__all__ = [
    "ZeroSumGame",
    "rps",
    "matching_pennies",
    "random_zero_sum",
    "exploitability",
    "methods",
    "train",
    "kuhn",
    "leduc",
]
