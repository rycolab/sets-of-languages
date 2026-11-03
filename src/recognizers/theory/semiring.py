from collections.abc import Callable, Sequence
from typing import Any

import torch

class Semiring[T]:
    r"""A semiring with custom addition and multiplication operations. It is
    assumed that elements are represented using :py:class:`~torch.Tensor`\ s.
    Conceptually, all operations act on multi-dimensional arrays of semiring
    elements, applying operations elementwise and allowing broadcasting.
    """

    def add(self, a: T, b: T) -> T:
        r"""Add two arrays of elements, allowing broadcasting."""
        raise NotImplementedError

    def add_in_place(self, a: T, b: T) -> None:
        r"""Add ``b`` to ``a`` in-place."""
        raise NotImplementedError

    def add_one_in_place(self, a: T) -> None:
        r"""Add :math:`\bm{1}` to ``a`` in-place."""
        self.add_in_place(a, self.ones((), self.get_dtype(a), self.get_device(a)))

    def sum(self, a: T, dims: tuple[int, ...]) -> T:
        r"""Sum over some dimensions of an array of elements."""
        raise NotImplementedError

    def multiply(self, a: T, b: T) -> T:
        r"""Multiply two arrays of elements, allowing broadcasting."""
        raise NotImplementedError

    def star(self, a: T) -> T:
        r"""Compute the star operation elementwise on an array of elements."""
        raise NotImplementedError

    def zeros(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> T:
        r"""Return an array of all zeros."""
        raise NotImplementedError

    def ones(self, size: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> T:
        r"""Return an array of all ones."""
        raise NotImplementedError

    def size(self, a: T) -> tuple[int, ...]:
        r"""Return the size of an array of elements."""
        raise NotImplementedError

    def transform_tensors(self, a: T, f: Callable[[torch.Tensor], torch.Tensor]) -> T:
        r"""Apply a function to all tensor objects that comprise an array of
        elements."""
        raise NotImplementedError

    def combine_tensors(self, values: Sequence[T], f: Callable[[Sequence[torch.Tensor]], torch.Tensor]) -> T:
        r"""Apply a function that combines tensor objects from different arrays
        of elements into a single array of elements."""
        raise NotImplementedError

    def set_index(self, a: T, index: Any, b: T) -> None:
        r"""Set a part of an array of elements to another array of elements."""
        raise NotImplementedError

    def equal(self, a: T, b: T) -> bool:
        r"""Check whether two arrays of elements are equal."""
        raise NotImplementedError

    def get_dtype(self, a: T) -> torch.dtype:
        r"""Get the dtype of this array of elements."""
        raise NotImplementedError

    def get_device(self, a: T) -> torch.device:
        r"""Get the device of this array of elements."""
        raise NotImplementedError

    def star_is_always_one(self) -> bool:
        r"""Whether the star operation always results in one. This can allow
        certain optimizations."""
        raise NotImplementedError

class TensorSemiring(Semiring[torch.Tensor]):
    r"""A semiring sub-class where arrays of elements are represented with a
    single :py:class:`~torch.Tensor` object. It is assumed that a fixed number
    of dimensions at the end of the tensor are reserved for semiring-specific
    purposes, e.g., if the elements are vectors rather than scalars. The rest of
    the dimensions at the beginning of the tensor are used to represent the size
    of the array of elements."""

    def transform_tensors(self, a: torch.Tensor, f: Callable[[torch.Tensor], torch.Tensor]):
        return f(a)

    def combine_tensors(self,
        values: Sequence[torch.Tensor],
        f: Callable[[Sequence[torch.Tensor]], torch.Tensor]
    ) -> torch.Tensor:
        return f(values)

    def set_index(self, a: torch.Tensor, index: Any, b: torch.Tensor) -> None:
        a[index] = b

    def equal(self, a: torch.Tensor, b: torch.Tensor) -> bool:
        return torch.allclose(a, b)

    def get_dtype(self, a: torch.Tensor) -> torch.dtype:
        return a.dtype

    def get_device(self, a: torch.Tensor) -> torch.device:
        return a.device
