import random

from recognizers.theory.dfa import (
    DeterministicFiniteAutomaton,
    DeterministicFiniteAutomatonContainer,
    State
)
from recognizers.theory.trim_nfa import trim_nfa
from recognizers.theory.minimize_dfa import minimize_trim_dfa
from recognizers.theory.dfa_accepts import dfa_accepts
from recognizers.tools.random_utils import (
    sample_from_negative_binomial,
    sample_from_bernoulli
)
from .density import has_max_density

def sample_podfa(
    mean_num_states: float,
    mean_alphabet_size: float,
    transition_probability: float,
    accept_probability: float,
    max_density: float,
    num_density_samples: int,
    density_length_range: tuple[int, int],
    generator: random.Random
) -> DeterministicFiniteAutomatonContainer:
    while True:
        num_states = sample_from_negative_binomial(mean_num_states, 1, generator)
        alphabet_size = sample_from_negative_binomial(mean_alphabet_size, 1, generator)
        M = DeterministicFiniteAutomatonContainer(
            num_states=num_states,
            alphabet_size=alphabet_size
        )
        for q in M.states():
            for a in M.alphabet():
                if sample_from_bernoulli(transition_probability, generator):
                    r = State(generator.randrange(q, num_states))
                    M.add_transition(q, a, r)
            if sample_from_bernoulli(accept_probability, generator):
                M.add_accept_state(q)
        trim_M = trim_nfa(M)
        # If the language of the DFA is trivial (empty set or all strings),
        # or finite (the DFA has no self-loops), reject it.
        # Also check that the density of the language is not too high.
        if (
            not is_trivial_dfa(trim_M) and
            is_infinite(trim_M) and
            has_max_density(
                lambda s: dfa_accepts(trim_M, s),
                max_density,
                num_density_samples,
                density_length_range,
                alphabet_size,
                generator
            )
        ):
            # Minimize the final DFA.
            return minimize_trim_dfa(trim_M)

def is_trivial_dfa(M: DeterministicFiniteAutomaton) -> bool:
    return accepts_no_strings(M) or accepts_all_strings(M)

def accepts_no_strings(M: DeterministicFiniteAutomaton) -> bool:
    # Because the DFA is trim, it's enough just to check a single state.
    return M.num_states() == 1 and not M.is_accept_state(M.start_state())

def accepts_all_strings(M: DeterministicFiniteAutomaton) -> bool:
    # Because the DFA is a trim PODFA with no cycles longer than 1 transition,
    # it is enough to check that all states accept, the DFA is complete (all
    # states have an outgoing transition on every symbol), and there is at least
    # one self-loop in the DFA.
    return (
        M.num_accept_states() == M.num_states() and
        M.is_complete() and
        any(t.state_from == t.state_to for t in M.transitions())
    )

def is_infinite(M: DeterministicFiniteAutomaton) -> bool:
    return any(t.state_from == t.state_to for t in M.transitions())
