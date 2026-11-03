import dataclasses
import math
import random
from collections.abc import Callable, Iterable, Sequence
from typing import cast, Literal

import numpy
import torch

from recognizers.theory.symbol import Symbol, ReservedSymbol
from recognizers.theory.automaton import State
from recognizers.theory.dfa import DeterministicFiniteAutomaton
from recognizers.theory.dfa_accepts import dfa_accepts
from recognizers.theory.nfa import NondeterministicFiniteAutomaton, SymbolOrEpsilon
from recognizers.theory.wdfa import WeightedDeterministicFiniteAutomaton
from recognizers.theory.wfa_allsum import wfa_backward
from recognizers.theory.log_binning_semiring import LogBinningSemiring
from recognizers.dataset_generation.weighted_language import (
    String,
    ValidNextSymbolSet,
    ValidNextSymbolList
)

@dataclasses.dataclass
class Transition:
    symbol: Symbol
    state_to: State
    log_weight: float

@dataclasses.dataclass
class TransitionDistribution:
    transitions: list[Transition]
    cumulative_weights: torch.Tensor

class WeightedDeterministicFiniteAutomatonSampler:

    _dfa: DeterministicFiniteAutomaton
    _transition_distributions: dict[State, TransitionDistribution]
    _accept_log_weights: dict[State, float]
    _next_symbols: list[ValidNextSymbolSet]
    _log_binning_allsum: torch.Tensor

    def __init__(self,
        dfa: DeterministicFiniteAutomaton,
        wdfa: WeightedDeterministicFiniteAutomaton[torch.Tensor],
        dtype: torch.dtype,
        device: torch.device
    ) -> None:
        super().__init__()
        self._dfa = dfa
        # Precompute the probabilities of sampling each transition conditioned
        # on the current state and target string length. Do this by performing
        # weight pushing in the log binning semiring over the WDFA (omitting the
        # step where we multiply the transition by the inverse). The first step
        # here is to compute the backward weights using Lehmann's algorithm.
        semiring = wdfa.semiring()
        if not isinstance(semiring, LogBinningSemiring):
            raise TypeError
        backward_log_binning_weights = wfa_backward(wdfa, dtype, device)
        # Next, group transitions by state_from and multiply the appropriate
        # backward weight into each transition weight.
        grouped_transitions: dict[State, list[tuple[Transition, torch.Tensor]]] = {}
        for wdfa_t in wdfa.transitions():
            grouped_transitions.setdefault(wdfa_t.state_from, []).append((
                Transition(
                    symbol=wdfa_t.symbol,
                    state_to=wdfa_t.state_to,
                    log_weight=wdfa_t.weight[1].item() if 1 < wdfa_t.weight.size(0) else -math.inf
                ),
                semiring.multiply(
                    wdfa_t.weight,
                    backward_log_binning_weights[wdfa_t.state_to]
                )
            ))
        # Now that the transitions are grouped by state_from and have had their
        # weights pushed, precompute the probability distributions over
        # transitions conditioned on each length by applying softmax to the log
        # weights, and precompute the cumulative sum for efficient sampling with
        # Random.choices().
        self._transition_distributions = {}
        for state_from, transition_list in grouped_transitions.items():
            self._transition_distributions[state_from] = TransitionDistribution(
                transitions=[t for t, w in transition_list],
                cumulative_weights=torch.cumsum(
                    # Note that if all of the log probabilities in a column are
                    # -inf, this will return nan. This is ok -- it indicates
                    # that there is no probability distribution at that index,
                    # because it is impossible to sample something from that
                    # index.
                    torch.softmax(
                        torch.stack([w for t, w in transition_list], dim=0),
                        dim=0
                    ),
                    dim=0
                ).cpu()
            )
        # Populate all the accept weights.
        self._accept_log_weights = {}
        for f, w in wdfa.accept_weights():
            self._accept_log_weights[f] = w[0].item()
        # Populate the next symbol lists for each state. This assumes that all
        # of the states in the DFA are co-accessible (can lead to an accept
        # state).
        self._next_symbols = [set() for _ in dfa.states()]
        for dfa_t in dfa.transitions():
            self._next_symbols[dfa_t.state_from].add(dfa_t.symbol)
        for f in dfa.accept_states():
            self._next_symbols[f].add(ReservedSymbol.EOS)
        # Store the sum of all path weights in the binning semiring, which is
        # the backward weight of the start state.
        self._log_binning_allsum = backward_log_binning_weights[dfa.start_state()].clone().cpu()

    def dfa(self) -> DeterministicFiniteAutomaton:
        return self._dfa

    def alphabet_size(self) -> int:
        return self._dfa.alphabet_size()

    def max_length(self) -> int:
        return self._log_binning_allsum.size(0) - 1

    def valid_lengths(self, length_range: tuple[int, int]) -> list[int]:
        lo, hi = length_range
        return [
            length
            for length, weight in zip(
                range(lo, hi+1),
                self._log_binning_allsum[lo:hi+1].tolist(),
                strict=True
            )
            if weight > -math.inf
        ]

    def is_valid_length(self, length: int) -> bool:
        return self._log_binning_allsum[length].item() > -math.inf

    def accepts(self, string: Iterable[Symbol]) -> bool:
        return dfa_accepts(self._dfa, string)

    def sample(self,
        length: int,
        generator: random.Random,
        include_log_probability: bool,
        include_next_symbols: bool
    ) -> tuple[String, float | None, ValidNextSymbolList | None]:
        sampled_string: list[Symbol] = []
        log_probability: float | None = 0.0 if include_log_probability else None
        next_symbols: ValidNextSymbolList | None = [] if include_next_symbols else None
        state = self._dfa.start_state()
        while True:
            if next_symbols is not None:
                # The set of next symbols is precomputed for each state.
                next_symbols.append(self._next_symbols[state])
            if length == 0:
                # All transitions scan exactly 1 symbol, so if we are at length
                # 0, the only action we can take is to accept.
                if log_probability is not None:
                    log_probability += self._accept_log_weights[state]
                return tuple(sampled_string), log_probability, next_symbols
            else:
                # We have at least 1 symbol left to scan. Sample the next
                # transition.
                transition_distribution = self._transition_distributions[state]
                cum_weights = transition_distribution.cumulative_weights[:, length]
                transition, = generator.choices(
                    transition_distribution.transitions,
                    # Convert the PyTorch tensor to a Numpy array to make it
                    # compatible with generator.choices(). It uses the same
                    # underlying storage, so there is no linear-time copy
                    # involved.
                    cum_weights=cast(Sequence[float], cum_weights.numpy())
                )
                if log_probability is not None:
                    log_probability += transition.log_weight
                sampled_string.append(transition.symbol)
                length -= 1
                state = transition.state_to

    def total_length_log_weight(self, length: int) -> float:
        return self._log_binning_allsum[length].item()
