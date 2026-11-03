import torch
from torch.testing import assert_close

from recognizers.theory.symbol import Symbol
from recognizers.theory.wfa_allsum import wfa_inner, wfa_backward, wfa_allsum
from recognizers.theory.wfa import WeightedFiniteAutomatonContainer
from recognizers.theory.wdfa import WeightedDeterministicFiniteAutomatonContainer
from recognizers.theory.binning_semiring import BinningSemiring

def test_lehmann_simple() -> None:
    semiring = BinningSemiring(4)
    dtype = torch.float32
    device = torch.device('cpu')
    M = WeightedFiniteAutomatonContainer[torch.Tensor](
        num_states=3,
        alphabet_size=2,
        semiring=semiring
    )
    q1, q2, q3 = M.states()
    a, b = map(Symbol, range(2))

    def w(values) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    M.set_transition_weight(q1, a, q3, w([0, 0.7, 0, 0]))
    M.set_transition_weight(q1, b, q2, w([0, 0.3, 0, 0]))
    M.set_transition_weight(q2, a, q3, w([0, 1, 0, 0]))
    M.set_accept_weight(q3, w([1, 0, 0, 0]))

    A = wfa_inner(M, dtype, device)
    assert A.size() == (3, 3, 4)
    assert_close(A, w([
        [[1, 0, 0, 0], [0, 0.3, 0, 0], [0, 0.7, 0.3, 0]],
        [[0, 0, 0, 0], [1, 0, 0, 0],   [0, 1, 0, 0]],
        [[0, 0, 0, 0], [0, 0, 0, 0],   [1, 0, 0, 0]]
    ]))

    d = wfa_backward(M, dtype, device)
    assert d.size() == (3, 4)
    assert_close(d, w([
        [0, 0.7, 0.3, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0]
    ]))

def test_backward_cycle() -> None:
    semiring = BinningSemiring(8)
    dtype = torch.float32
    device = torch.device('cpu')
    M = WeightedFiniteAutomatonContainer[torch.Tensor](
        num_states=3,
        alphabet_size=3,
        semiring=semiring
    )
    q1, q2, q3 = M.states()
    a, b, c = map(Symbol, range(3))

    def w(values) -> torch.Tensor:
        return torch.tensor(values, dtype=dtype, device=device)

    M.set_transition_weight(q1, a, q2, w([0, 1, 0, 0, 0, 0, 0, 0]))
    M.set_transition_weight(q2, b, q3, w([0, 0.1, 0, 0, 0, 0, 0, 0]))
    M.set_transition_weight(q3, a, q2, w([0, 1, 0, 0, 0, 0, 0, 0]))
    M.set_accept_weight(q2, w([0.9, 0, 0, 0, 0, 0, 0, 0]))

    backward_weights = wfa_backward(M, dtype, device)
    assert backward_weights.size() == (3, 8)
    assert_close(backward_weights, w([
        [0, 0.9, 0, 0.09, 0, 0.009, 0, 0.0009],
        [0.9, 0, 0.09, 0, 0.009, 0, 0.0009, 0],
        [0, 0.9, 0, 0.09, 0, 0.009, 0, 0.0009]
    ]))

def test_wdfa_no_accept_states() -> None:
    semiring = BinningSemiring(8)
    dtype = torch.float32
    device = torch.device('cpu')
    M = WeightedDeterministicFiniteAutomatonContainer[torch.Tensor](
        alphabet_size=2,
        semiring=semiring
    )
    allsum = wfa_allsum(M, dtype, device)
    assert allsum.size() == (8,)
    assert_close(allsum, torch.zeros((8,), dtype=dtype, device=device))

def test_wfa_no_states() -> None:
    semiring = BinningSemiring(8)
    dtype = torch.float32
    device = torch.device('cpu')
    M = WeightedFiniteAutomatonContainer[torch.Tensor](
        alphabet_size=2,
        semiring=semiring
    )
    allsum = wfa_allsum(M, dtype, device)
    assert allsum.size() == (8,)
    assert_close(allsum, torch.zeros((8,), dtype=dtype, device=device))
