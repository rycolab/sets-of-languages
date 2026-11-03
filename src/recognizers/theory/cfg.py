import dataclasses
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast, NewType

from orderedsets import OrderedSet

from .grammar import Variable, Terminal, Grammar, GrammarContainer

@dataclasses.dataclass(frozen=True)
class Rule:
    r"""A rule in a CFG."""
    left: Variable
    right: tuple[Variable | Terminal, ...]

class ContextFreeGrammar(Grammar):
    r"""A context-free grammar (CFG)."""

    def rules(self) -> Iterable[Rule]:
        r"""Get the set of rules."""
        raise NotImplementedError

    def num_rules(self) -> int:
        r"""Get the number of rules."""
        return sum(1 for _ in self.rules())

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, ContextFreeGrammar) and
            self.num_variables() == other.num_variables() and
            self.num_terminals() == other.num_terminals() and
            set(self.rules()) == set(other.rules()) and
            self.start_variable() == other.start_variable()
        )

class ContextFreeGrammarContainer(GrammarContainer, ContextFreeGrammar):

    _rules: OrderedSet[Rule]

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
        self._rules = OrderedSet()

    @staticmethod
    def from_parts(
        num_variables: int,
        num_terminals: int,
        rules: Callable[
            [Sequence[Variable], Sequence[Terminal]],
            Iterable[tuple[Variable, tuple[Variable | Terminal, ...]]]
        ]
    ) -> ContextFreeGrammarContainer:
        result = ContextFreeGrammarContainer(
            num_variables=num_variables,
            num_terminals=num_terminals
        )
        result.add_rules_from_parts(rules(result.variables(), result.terminals()))
        return result

    def add_rules_from_parts(self,
        rules: Iterable[tuple[Variable, tuple[Variable | Terminal, ...]]]
    ) -> None:
        for left, right in rules:
            self.add_rule(left, right)

    def rules(self) -> Iterable[Rule]:
        return self._rules

    def num_rules(self) -> int:
        return len(self._rules)

    def add_rule(self, left: Variable, right: tuple[Variable | Terminal, ...]) -> None:
        self._rules.add(Rule(left, right))
