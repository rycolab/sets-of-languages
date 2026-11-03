import random

import torch

from recognizers.theory.automaton import State
from recognizers.theory.dfa import DeterministicFiniteAutomatonContainer
from recognizers.theory.minimize_dfa import minimize_dfa
from recognizers.theory.min_dfa_is_star_free import min_dfa_is_star_free
from recognizers.theory.dfa_accepts import dfa_accepts
from recognizers.tools.random_utils import (
    sample_from_negative_binomial,
    sample_from_bernoulli
)
from .density import has_max_density

def sample_dfa(
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
                    r = State(generator.randrange(num_states))
                    M.add_transition(q, a, r)
            if sample_from_bernoulli(accept_probability, generator):
                M.add_accept_state(q)
        # Minimize the DFA and check if its language is star-free.
        # Also check that the density of the language is not too high.
        min_M = minimize_dfa(M)
        if (
            not min_dfa_is_star_free(min_M) and
            has_max_density(
                lambda s: dfa_accepts(min_M, s),
                max_density,
                num_density_samples,
                density_length_range,
                alphabet_size,
                generator
            )
        ):
            return min_M
