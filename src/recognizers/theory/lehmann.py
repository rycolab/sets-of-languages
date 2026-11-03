import torch

from .semiring import Semiring

def lehmann[T](A: T, semiring: Semiring[T]) -> None:
    num_states, other_num_states = semiring.size(A)
    if other_num_states != num_states:
        raise ValueError('input must be a square matrix')
    for k in range(num_states):
        semiring.add_in_place(
            A,
            semiring.multiply(
                semiring.multiply(
                    semiring.transform_tensors(A, lambda x: x[:, k, None]),
                    semiring.transform_tensors(
                        semiring.star(semiring.transform_tensors(A, lambda x: x[k, k])),
                        lambda x: x[None, None]
                    )
                ),
                semiring.transform_tensors(A, lambda x: x[k, None, :])
            )
        )
    semiring.add_one_in_place(
        semiring.transform_tensors(A, lambda x: move_last_dim(torch.diagonal(x)))
    )

def floyd_warshall[T](A: T, semiring: Semiring[T]) -> None:
    num_states, other_num_states = semiring.size(A)
    if other_num_states != num_states:
        raise ValueError('input must be a square matrix')
    for k in range(num_states):
        semiring.add_in_place(
            A,
            semiring.multiply(
                semiring.transform_tensors(A, lambda x: x[:, k, None]),
                semiring.transform_tensors(A, lambda x: x[k, None, :])
            )
        )
    semiring.add_one_in_place(
        semiring.transform_tensors(A, lambda x: move_last_dim(torch.diagonal(x)))
    )

def move_last_dim(x: torch.Tensor) -> torch.Tensor:
    if x.ndim > 1:
        last_dim = x.ndim - 1
        permutation = (last_dim, *range(last_dim))
        x = x.permute(*permutation)
    return x
