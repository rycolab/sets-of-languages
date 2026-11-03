import math
from collections.abc import Callable

import torch

from recognizers.theory.automaton import State
from recognizers.theory.symbol import ReservedSymbol
from recognizers.theory.nfa import Transition as NFATransition
from recognizers.theory.dfa import (
    DeterministicFiniteAutomaton,
    Transition as DFATransition
)
from recognizers.theory.wdfa import WeightedDeterministicFiniteAutomatonContainer
from recognizers.theory.log_binning_semiring import LogBinningSemiring
from recognizers.string_sampling.sampler_util import count_to_bins

def lift_dfa_weights(
    M: DeterministicFiniteAutomaton,
    *,
    max_length: int,
    dtype: torch.dtype,
    device: torch.device
) -> WeightedDeterministicFiniteAutomatonContainer[torch.Tensor]:
    r"""Given a partial DFA, assign uniform probabilities to make it a PDFA,
    then lift it to the log binning semiring.
    """
    weight_size = max_length + 1
    result = WeightedDeterministicFiniteAutomatonContainer(
        num_states=M.num_states(),
        alphabet_size=M.alphabet_size(),
        start_state=M.start_state(),
        semiring=LogBinningSemiring(weight_size)
    )
    grouped_transitions: list[list[DFATransition]] = [[] for q in M.states()]
    for t in M.transitions():
        grouped_transitions[t.state_from].append(t)
    for state_from_int, transitions in enumerate(grouped_transitions):
        state_from = State(state_from_int)
        is_accept_state = M.is_accept_state(state_from)
        num_actions = len(transitions) + int(is_accept_state)
        if num_actions > 0:
            log_prob = -math.log(num_actions)
            for t in transitions:
                result.set_transition_weight(
                    t.state_from,
                    t.symbol,
                    t.state_to,
                    count_to_bins(1, log_prob, weight_size, dtype, device)
                )
            if is_accept_state:
                result.set_accept_weight(
                    state_from,
                    count_to_bins(0, log_prob, weight_size, dtype, device)
                )
    return result
