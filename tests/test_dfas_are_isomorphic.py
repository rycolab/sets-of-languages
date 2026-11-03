import random

from recognizers.theory.symbol import Symbol
from recognizers.theory.automaton import State
from recognizers.theory.dfa import DeterministicFiniteAutomaton, DeterministicFiniteAutomatonContainer
from recognizers.theory.dfas_are_isomorphic import dfas_are_isomorphic
from recognizers.theory.trim_nfa import trim_nfa
from random_dfa import random_dfa, permute_dfa

def permute_states(
    M: DeterministicFiniteAutomaton,
    permutation: list[State]
) -> DeterministicFiniteAutomatonContainer:
    R = DeterministicFiniteAutomatonContainer(
        num_states=M.num_states(),
        alphabet_size=M.alphabet_size(),
        start_state=permutation[M.start_state()]
    )
    for t in M.transitions():
        R.add_transition(permutation[t.state_from], t.symbol, permutation[t.state_to])
    for f in M.accept_states():
        R.add_accept_state(permutation[f])
    return R

def test_accept_all() -> None:
    M = DeterministicFiniteAutomatonContainer.from_parts(1, 2, 0, [(0, 0, 0), (0, 1, 0)], [0])
    assert dfas_are_isomorphic(M, M)

def test_accept_none() -> None:
    M = DeterministicFiniteAutomatonContainer.from_parts(1, 2, 0, [], [])
    assert dfas_are_isomorphic(M, M)

def remove_random_transition(
    M: DeterministicFiniteAutomatonContainer,
    generator: random.Random
) -> None:
    t = generator.choice(list(M.transitions()))
    M.remove_transition(t.state_from, t.symbol)

def change_random_transition(
    M: DeterministicFiniteAutomatonContainer,
    generator: random.Random
) -> None:
    t = generator.choice(list(M.transitions()))
    new_state_to = generator.choice([q for q in M.states() if q != t.state_to])
    M.set_transition(t.state_from, t.symbol, new_state_to)

def test_random_dfas() -> None:
    generator = random.Random(123)
    for i in range(100):
        M = trim_nfa(random_dfa(
            mean_num_states=10,
            mean_alphabet_size=5,
            transition_probability=0.6,
            accept_probability=0.4,
            generator=generator
        ))
        # Test that a DFA with permuted states is still isomorphic.
        permuted_M = permute_dfa(M, generator)
        assert dfas_are_isomorphic(M, permuted_M)
        assert dfas_are_isomorphic(permuted_M, M)
        # Test that a DFA with deleted transitions is not isomorphic.
        if M.num_transitions() > 0:
            modified_M = permuted_M.copy()
            for _ in range(generator.randrange(M.num_transitions()) + 1):
                remove_random_transition(modified_M, generator)
            assert not dfas_are_isomorphic(M, modified_M)
            assert not dfas_are_isomorphic(modified_M, M)
        # Test that a DFA with the same number of transitions but transitions
        # that go to different states is still isomorphic.
        if M.num_transitions() > 0 and M.num_states() >= 2 and M.alphabet_size() >= 2:
            modified_M = permuted_M.copy()
            change_random_transition(modified_M, generator)
            trim_modified_M = trim_nfa(modified_M)
            assert not dfas_are_isomorphic(M, trim_modified_M)
            assert not dfas_are_isomorphic(trim_modified_M, M)
