from collections.abc import Iterable

from .symbol import Symbol
from .dfa import DeterministicFiniteAutomaton

def dfa_accepts(M: DeterministicFiniteAutomaton, w: Iterable[Symbol]) -> bool:
    q = M.start_state()
    for wi in w:
        next_q = M.transition(q, wi)
        if next_q is None:
            return False
        q = next_q
    return M.is_accept_state(q)
