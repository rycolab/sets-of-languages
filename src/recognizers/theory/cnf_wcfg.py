import itertools
from collections.abc import Iterable

from .grammar import Variable, Terminal, GrammarContainer
from .wcfg import (
    WeightedContextFreeGrammar,
    Rule,
    SemiringMixin
)
from .cnf_cfg import (
    EmptyRule as CFGEmptyRule,
    BinaryRule as CFGBinaryRule,
    TerminalRule as CFGTerminalRule
)
from .semiring import Semiring

class EmptyRule[Weight](CFGEmptyRule, Rule[Weight]):

    def __init__(self, left: Variable, weight: Weight) -> None:
        Rule.__init__(self, left, (), weight)

class BinaryRule[Weight](CFGBinaryRule, Rule[Weight]):

    def __init__(self,
        left: Variable,
        first: Variable,
        second: Variable,
        weight: Weight
    ) -> None:
        Rule.__init__(self, left, (first, second), weight)

class TerminalRule[Weight](CFGTerminalRule, Rule[Weight]):

    def __init__(self,
        left: Variable,
        terminal: Terminal,
        weight: Weight
    ) -> None:
        Rule.__init__(self, left, (terminal,), weight)

class ChomskyNormalFormWeightedContextFreeGrammar[Weight](
    WeightedContextFreeGrammar[Weight]
):

    def rules(self) -> Iterable[Rule]:
        empty_rule = self.empty_rule()
        return itertools.chain(
            [empty_rule] if empty_rule is not None else [],
            self.binary_rules(),
            self.terminal_rules()
        )

    def empty_rule_weight(self) -> Weight | None:
        raise NotImplementedError

    def empty_rule(self) -> EmptyRule | None:
        weight = self.empty_rule_weight()
        if weight is not None:
            return EmptyRule(self.start_variable(), weight)
        else:
            return None

    def binary_rules(self) -> Iterable[BinaryRule]:
        raise NotImplementedError

    def terminal_rules(self) -> Iterable[TerminalRule]:
        raise NotImplementedError

class ChomskyNormalFormWeightedContextFreeGrammarContainer[Weight](
    SemiringMixin,
    GrammarContainer,
    ChomskyNormalFormWeightedContextFreeGrammar[Weight]
):

    _empty_rule_weight: Weight | None
    _binary_rule_weights: dict[tuple[Variable, Variable, Variable], Weight]
    _terminal_rule_weights: dict[tuple[Variable, Terminal], Weight]

    def __init__(self,
        *,
        num_variables: int = 1,
        num_terminals: int,
        start_variable: Variable | None = None,
        semiring: Semiring[Weight]
    ) -> None:
        super().__init__(
            num_variables=num_variables,
            num_terminals=num_terminals,
            start_variable=start_variable,
            semiring=semiring
        )
        self._empty_rule_weight = None
        self._binary_rule_weights = {}
        self._terminal_rule_weights = {}

    def empty_rule_weight(self) -> Weight | None:
        return self._empty_rule_weight

    def set_empty_rule_weight(self, weight: Weight) -> None:
        self._empty_rule_weight = weight

    def binary_rules(self) -> Iterable[BinaryRule]:
        for (left, first, second), weight in self._binary_rule_weights.items():
            yield BinaryRule(left, first, second, weight)

    def set_binary_rule_weight(self,
        left: Variable,
        first: Variable,
        second: Variable,
        weight: Weight
    ) -> None:
        total = self._num_terminals + self._num_variables
        if not (self._num_terminals <= left < total):
            raise ValueError(f'left side ({left}) must be a valid variable')
        if not (self._num_terminals <= first < total):
            raise ValueError(f'first symbol on right side ({first}) must be a valid variable')
        if not (self._num_terminals <= second < total):
            raise ValueError(f'second symbol on right side ({second}) must be a valid variable')
        if self._start_variable in (first, second):
            raise ValueError(f'the start variable cannot be on the right side of a rule')
        self._binary_rule_weights[left, first, second] = weight

    def terminal_rules(self) -> Iterable[TerminalRule]:
        for (left, terminal), weight in self._terminal_rule_weights.items():
            yield TerminalRule(left, terminal, weight)

    def set_terminal_rule_weight(self,
        left: Variable,
        terminal: Terminal,
        weight: Weight
    ) -> None:
        if not (self._num_terminals <= left < self._num_terminals + self._num_variables):
            raise ValueError(f'left side ({left}) must be a valid variable')
        if not (0 <= terminal < self._num_terminals):
            raise ValueError(f'invalid terminal {terminal}')
        self._terminal_rule_weights[left, terminal] = weight
