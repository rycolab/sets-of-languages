import random

from recognizers.theory.symbol import Symbol, ReservedSymbol
from recognizers.theory.nfa import NondeterministicFiniteAutomatonContainer
from recognizers.theory.nfa_accepts import nfa_accepts
from random_string import random_string

def test_example() -> None:
    # This NFA recognizes the language |(0|1|)1+ = |0?1+.
    M = NondeterministicFiniteAutomatonContainer.from_parts(
        4,
        2,
        0,
        [
            (0, 0, 1),
            (0, ReservedSymbol.EPSILON, 1),
            (0, 0, 2),
            (0, 1, 2),
            (1, ReservedSymbol.EPSILON, 2),
            (2, 1, 1),
            (2, 1, 3)
        ],
        [0, 3]
    )
    def should_accept(w: list[Symbol]) -> bool:
        if len(w) > 0:
            if w[0] == 0:
                w = w[1:]
            return len(w) >= 1 and all(wi == 1 for wi in w)
        else:
            return True
    generator = random.Random(123)
    for i in range(100):
        w = random_string(M.alphabet_size(), 0, 20, generator)
        assert nfa_accepts(M, w) == should_accept(w)
