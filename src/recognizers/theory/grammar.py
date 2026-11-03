import dataclasses
from collections.abc import Iterable, Sequence
from typing import Any, cast, NewType

from .symbol import Symbol

Terminal = NewType('Terminal', Symbol)
Variable = NewType('Variable', int)
ZeroIndexedVariable = NewType('ZeroIndexedVariable', int)

class Grammar:
    r"""A base class for grammars.

    Variables and terminals are represented as ints from 0 to ``num_terminals +
    num_variables - 1``. Terminals are always numbered from 0 to ``num_terminals
    - 1``, and variables are always numbered from ``num_terminals`` to
    ``num_terminals + num_variables - 1``.
    """

    def variables(self) -> Sequence[Variable]:
        r"""Get the set of variables (nonterminals)."""
        num_terminals = self.num_terminals()
        return cast(Sequence[Variable], range(num_terminals, num_terminals + self.num_variables()))

    def num_variables(self) -> int:
        r"""Get the number of variables."""
        raise NotImplementedError

    def is_variable(self, symbol: Terminal | Variable) -> bool:
        r"""Tell whether a symbol is a variable."""
        return self.num_terminals() <= symbol

    def zero_indexed(self, variable: Variable) -> ZeroIndexedVariable:
        r"""Subtract the number of terminals from a variable to make it
        zero-indexed.
        """
        return ZeroIndexedVariable(variable - self.num_terminals())

    def terminals(self) -> Sequence[Terminal]:
        r"""Get the set of terminals."""
        return cast(Sequence[Terminal], range(self.num_terminals()))

    def num_terminals(self) -> int:
        r"""Get the number of terminals."""
        raise NotImplementedError

    def is_terminal(self, symbol: Terminal | Variable) -> bool:
        r"""Tell whether a symbol is a terminal."""
        return symbol < self.num_terminals()

    def start_variable(self) -> Variable:
        r"""Get the start variable."""
        raise NotImplementedError

class GrammarContainer(Grammar):

    _num_variables: int
    _num_terminals: int
    _start_variable: Variable

    def __init__(self,
        *,
        num_variables: int = 1,
        num_terminals: int,
        start_variable: Variable | None = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        if not (1 <= num_variables):
            raise ValueError('num_variables must be at least 1')
        self._num_variables = num_variables
        if not (0 <= num_terminals):
            raise ValueError('num_terminals must be at least 0')
        self._num_terminals = num_terminals
        if start_variable is None:
            start_variable = Variable(num_terminals)
        if not (num_terminals <= start_variable < num_terminals + num_variables):
            raise ValueError(f'invalid start variable {start_variable}')
        self._start_variable = start_variable

    def num_variables(self) -> int:
        return self._num_variables

    def num_terminals(self) -> int:
        return self._num_terminals

    def start_variable(self) -> Variable:
        return self._start_variable
