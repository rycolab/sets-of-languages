import math
from collections.abc import Callable, Iterable

from orderedsets import OrderedSet

from recognizers.tools.graph_traversal import graph_vertexes
from .automaton import State
from .dfa import DeterministicFiniteAutomaton
from .complete_dfa import CompleteDeterministicFiniteAutomaton, make_dfa_complete

def min_dfa_is_star_free(M: DeterministicFiniteAutomaton) -> bool:
    r"""
    :param M: A minimal DFA. The DFA must be minimal for the output to be
        correct.
    """
    return is_aperiodic(syntactic_monoid(make_dfa_complete(M)))

def compose_tuples(a: tuple[State, ...], b: tuple[State, ...]) -> tuple[State, ...]:
    return tuple(b[a[q]] for q in range(len(a)))

def syntactic_monoid(M: CompleteDeterministicFiniteAutomaton) -> set[tuple[State, ...]]:
    r"""
    :param M: A complete minimal DFA. The DFA must be minimal for the output to
        be correct.
    """
    identity = tuple(M.states())
    symbol_transformations = list(OrderedSet(
        tuple(M.transition(q, a) for q in M.states())
        for a in M.alphabet()
    ))
    def successor_func(u: tuple[State, ...]) -> Iterable[tuple[State, ...]]:
        for t in symbol_transformations:
            yield compose_tuples(u, t)
    return graph_vertexes([identity], successor_func)

def is_aperiodic(syntactic_monoid: set[tuple[State, ...]]) -> bool:
    # For each transformation t, check whether t^(i-1) == t^i for some i >= 1.
    # Let n be len(syntactic_monoid). We know that i = n+1 is the highest power
    # we need to check, because n is the number of distinct transformations
    # possible; if it hasn't stabilized by then, it never will. Note that after
    # the least i where this is true, i', it is true for all i >= i'. We can
    # reach this point faster using exponentiation by squaring, in which case we
    # need at most ceil(log_2(n)) multiplications to compute t^(n') for some
    # n' >= n, then one more multiplication to check t^(n') == t^(n'+1).
    num_iters = math.ceil(math.log2(len(syntactic_monoid)))
    return all(transformation_is_aperiodic(t, num_iters) for t in syntactic_monoid)

def transformation_is_aperiodic(t: tuple[State, ...], num_iters: int) -> bool:
    power = t
    for _ in range(num_iters):
        power = compose_tuples(power, power)
    return power == compose_tuples(power, t)
