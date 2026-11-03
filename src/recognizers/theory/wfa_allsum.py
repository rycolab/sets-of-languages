from collections.abc import Callable, Sequence

import torch

from .wfa import WeightedFiniteAutomaton
from .wdfa import WeightedDeterministicFiniteAutomaton
from .lehmann import lehmann, floyd_warshall

def wfa_inner[T](
    M: WeightedFiniteAutomaton[T],
    dtype: torch.dtype,
    device: torch.device
) -> T:
    semiring = M.semiring()
    num_states = M.num_states()
    inner_weights = semiring.zeros(size=(num_states, num_states), dtype=dtype, device=device)
    for t in M.transitions():
        # Coalesce weights over different symbols.
        semiring.add_in_place(
            semiring.transform_tensors(inner_weights, lambda x: x[t.state_from, t.state_to]),
            t.weight
        )
    if semiring.star_is_always_one():
        floyd_warshall(inner_weights, semiring)
    else:
        lehmann(inner_weights, semiring)
    return inner_weights

def wfa_backward_from_inner[T](
    M: WeightedFiniteAutomaton[T],
    inner_weights: T,
    dtype: torch.dtype,
    device: torch.device
) -> T:
    semiring = M.semiring()
    accept_states = []
    accept_weight_list = []
    for f, w in M.accept_weights():
        accept_states.append(f)
        accept_weight_list.append(w)
    if accept_states:
        accept_weights = semiring.combine_tensors(accept_weight_list, _ensure_list(torch.stack))
        return semiring.sum(
            semiring.multiply(
                # We only need to use the inner weights for the accept states.
                semiring.transform_tensors(inner_weights, lambda x: x[:, accept_states]),
                semiring.transform_tensors(accept_weights, lambda x: x[None])
            ),
            dims=(1,)
        )
    else:
        return semiring.zeros((M.num_states(),), dtype, device)

def wfa_backward[T](
    M: WeightedFiniteAutomaton[T],
    dtype: torch.dtype,
    device: torch.device
) -> T:
    return wfa_backward_from_inner(M, wfa_inner(M, dtype, device), dtype, device)

def wfa_allsum[T](
    M: WeightedFiniteAutomaton[T],
    dtype: torch.dtype,
    device: torch.device
) -> T:
    semiring = M.semiring()
    inner_weights = wfa_inner(M, dtype, device)
    if isinstance(M, WeightedDeterministicFiniteAutomaton):
        # WDFAs have a single start state with weight one. In this case, we can
        # just return the backward weight of the start state. We can skip
        # multiplication with the start weight and summation over start states.
        q = M.start_state()
        backward_weights = wfa_backward_from_inner(
            M,
            # We only need to use the inner weights for the start state.
            semiring.transform_tensors(inner_weights, lambda x: x[[q]]),
            dtype,
            device
        )
        return semiring.transform_tensors(backward_weights, lambda x: x[0])
    else:
        start_states = []
        start_weight_list = []
        for q, w in M.start_weights():
            start_states.append(q)
            start_weight_list.append(w)
        if start_states:
            start_weights = semiring.combine_tensors(start_weight_list, _ensure_list(torch.stack))
            backward_weights = wfa_backward_from_inner(
                M,
                # We only need to use the inner weights for the start states.
                semiring.transform_tensors(inner_weights, lambda x: x[start_states]),
                dtype,
                device
            )
            return semiring.sum(semiring.multiply(start_weights, backward_weights), dims=(0,))
        else:
            return semiring.zeros((), dtype, device)

def _ensure_list[T](func: Callable[[tuple[T, ...] | list[T]], T]) -> Callable[[Sequence[T]], T]:
    def result(values: Sequence[T]) -> T:
        return func(values if isinstance(values, (tuple, list)) else tuple(values))
    return result
