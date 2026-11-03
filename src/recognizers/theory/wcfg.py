import dataclasses
from collections.abc import Iterable

from .grammar import Grammar, Variable, Terminal, GrammarContainer
from .cfg import Rule as CFGRule
from .semiring import Semiring

class Rule[Weight](CFGRule):

    weight: Weight

    def __init__(self,
        left: Variable,
        right: tuple[Variable | Terminal, ...],
        weight: Weight
    ) -> None:
        super().__init__(left, right)
        self.weight = weight

class WeightedContextFreeGrammar[Weight](Grammar):
    r"""A weighted context-free grammar (WCFG).

    The weights come from a :py:class:`Semiring`.
    """

    def semiring(self) -> Semiring[Weight]:
        r"""Get the semiring which the weights of this WCFG come from."""
        raise NotImplementedError

    def rules(self) -> Iterable[Rule]:
        raise NotImplementedError

    def rule_weight(self,
        left: Variable,
        right: tuple[Variable | Terminal, ...]
    ) -> Weight | None:
        raise NotImplementedError

class SemiringMixin[Weight](GrammarContainer):

    _semiring: Semiring[Weight]

    def __init__(self, *, semiring: Semiring, **kwargs) -> None:
        super().__init__(**kwargs)
        self._semiring = semiring

    def semiring(self) -> Semiring[Weight]:
        return self._semiring
