import dataclasses
import itertools
from collections.abc import Callable, Iterable, Sequence
from typing import cast

from orderedsets import OrderedSet

from .grammar import (
    Variable,
    Terminal,
    GrammarContainer
)
from .cfg import (
    ContextFreeGrammar,
    Rule
)

class EmptyRule(Rule):

    def __init__(self, left: Variable) -> None:
        super().__init__(left, ())

class BinaryRule(Rule):

    right: tuple[Variable, Variable]

    def __init__(self, left: Variable, first: Variable, second: Variable) -> None:
        super().__init__(left, (first, second))

    @property
    def first(self) -> Variable:
        return self.right[0]

    @property
    def second(self) -> Variable:
        return self.right[1]

class TerminalRule(Rule):

    right: tuple[Terminal]

    def __init__(self, left: Variable, terminal: Terminal) -> None:
        super().__init__(left, (terminal,))

    @property
    def terminal(self) -> Terminal:
        return self.right[0]

class ChomskyNormalFormContextFreeGrammar(ContextFreeGrammar):
    r"""A context-free grammar in Chomsky normal form (CNF)."""

    def rules(self) -> Iterable[EmptyRule | BinaryRule | TerminalRule]:
        empty_rule = self.empty_rule()
        return itertools.chain(
            [empty_rule] if empty_rule is not None else [],
            self.binary_rules(),
            self.terminal_rules()
        )

    def num_rules(self) -> int:
        return (
            int(self.has_empty_rule()) +
            self.num_binary_rules() +
            self.num_terminal_rules()
        )

    def has_empty_rule(self) -> bool:
        raise NotImplementedError

    def empty_rule(self) -> EmptyRule | None:
        if self.has_empty_rule():
            return EmptyRule(self.start_variable())
        else:
            return None

    def binary_rules(self) -> Iterable[BinaryRule]:
        raise NotImplementedError

    def num_binary_rules(self) -> int:
        return sum(1 for _ in self.binary_rules())

    def terminal_rules(self) -> Iterable[TerminalRule]:
        raise NotImplementedError

    def num_terminal_rules(self) -> int:
        return sum(1 for _ in self.terminal_rules())

class ChomskyNormalFormContextFreeGrammarContainer(
    GrammarContainer,
    ChomskyNormalFormContextFreeGrammar
):

    _has_empty_rule: bool
    _binary_rules: OrderedSet[BinaryRule]
    _terminal_rules: OrderedSet[TerminalRule]

    def __init__(self,
        *,
        num_variables: int = 1,
        num_terminals: int,
        start_variable: Variable | None = None
    ) -> None:
        super().__init__(
            num_variables=num_variables,
            num_terminals=num_terminals,
            start_variable=start_variable
        )
        self._has_empty_rule = False
        self._binary_rules = OrderedSet()
        self._terminal_rules = OrderedSet()

    @staticmethod
    def from_parts(
        num_variables: int,
        num_terminals: int,
        rules: Callable[
            [Sequence[Variable], Sequence[Terminal]],
            Iterable[
                tuple[Variable,] |
                tuple[Variable, Terminal] |
                tuple[Variable, Variable, Variable]
            ]
        ]
    ) -> ChomskyNormalFormContextFreeGrammarContainer:
        result = ChomskyNormalFormContextFreeGrammarContainer(
            num_variables=num_variables,
            num_terminals=num_terminals
        )
        result.add_rules_from_parts(rules(result.variables(), result.terminals()))
        return result

    def add_rules_from_parts(self,
        rules: Iterable[
            tuple[Variable,] |
            tuple[Variable, Terminal] |
            tuple[Variable, Variable, Variable]
        ]
    ) -> None:
        for rule in rules:
            match rule:
                case (A,):
                    if A == self.start_variable():
                        self.add_empty_rule()
                    else:
                        raise ValueError
                case (A, a):
                    self.add_terminal_rule(cast(Variable, A), cast(Terminal, a))
                case (A, B, C):
                    self.add_binary_rule(cast(Variable, A), cast(Variable, B), cast(Variable, C))
                case _:
                    raise TypeError

    def has_empty_rule(self) -> bool:
        return self._has_empty_rule

    def add_empty_rule(self) -> None:
        self._has_empty_rule = True

    def binary_rules(self) -> Iterable[BinaryRule]:
        return self._binary_rules

    def num_binary_rules(self) -> int:
        return len(self._binary_rules)

    def add_binary_rule(self, left: Variable, first: Variable, second: Variable) -> None:
        total = self._num_terminals + self._num_variables
        if not (self._num_terminals <= left < total):
            raise ValueError(f'left side ({left}) must be a valid variable')
        if not (self._num_terminals <= first < total):
            raise ValueError(f'first symbol on right side ({first}) must be a valid variable')
        if not (self._num_terminals <= second < total):
            raise ValueError(f'second symbol on right side ({second}) must be a valid variable')
        if self._start_variable in (first, second):
            raise ValueError(f'the start variable cannot be on the right side of a rule')
        self._binary_rules.add(BinaryRule(left, first, second))

    def terminal_rules(self) -> Iterable[TerminalRule]:
        return self._terminal_rules

    def num_terminal_rules(self) -> int:
        return len(self._terminal_rules)

    def add_terminal_rule(self, left: Variable, terminal: Terminal) -> None:
        if not (self._num_terminals <= left < self._num_terminals + self._num_variables):
            raise ValueError(f'left side ({left}) must be a valid variable')
        if not (0 <= terminal < self._num_terminals):
            raise ValueError(f'invalid terminal {terminal}')
        self._terminal_rules.add(TerminalRule(left, terminal))
