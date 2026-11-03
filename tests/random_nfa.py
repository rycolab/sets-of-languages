import itertools
import random
from collections.abc import Iterable
from typing import cast

from recognizers.theory.nfa import (
    NondeterministicFiniteAutomaton,
    NondeterministicFiniteAutomatonContainer,
    SymbolOrEpsilon
)
from recognizers.theory.symbol import ReservedSymbol
from recognizers.theory.automaton import State
from recognizers.tools.random_utils import (
    sample_from_negative_binomial,
    sample_from_bernoulli
)

def random_nfa(
    mean_num_states: float,
    mean_alphabet_size: float,
    transition_probability: float,
    accept_probability: float,
    generator: random.Random
) -> NondeterministicFiniteAutomatonContainer:
    num_states = sample_from_negative_binomial(mean_num_states, 1, generator)
    alphabet_size = sample_from_negative_binomial(mean_alphabet_size, 1, generator)
    M = NondeterministicFiniteAutomatonContainer(
        num_states=num_states,
        alphabet_size=alphabet_size
    )
    for q in M.states():
        for a in cast(Iterable[SymbolOrEpsilon], itertools.chain(M.alphabet(), [ReservedSymbol.EPSILON])):
            for r in M.states():
                if sample_from_bernoulli(transition_probability, generator):
                    M.add_transition(q, a, r)
        if sample_from_bernoulli(accept_probability, generator):
            M.add_accept_state(q)
    return M

def permute_nfa(
    M: NondeterministicFiniteAutomaton,
    generator: random.Random
) -> NondeterministicFiniteAutomatonContainer:
    permutation = list(M.states())
    generator.shuffle(permutation)
    result = NondeterministicFiniteAutomatonContainer(
        num_states=M.num_states(),
        alphabet_size=M.alphabet_size(),
        start_state=permutation[M.start_state()]
    )
    transitions = list(M.transitions())
    generator.shuffle(transitions)
    for t in transitions:
        result.add_transition(permutation[t.state_from], t.symbol, permutation[t.state_to])
    accept_states = list(M.accept_states())
    generator.shuffle(accept_states)
    for f in accept_states:
        result.add_accept_state(permutation[f])
    return result
