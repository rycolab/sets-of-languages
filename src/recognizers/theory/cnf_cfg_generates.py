from collections.abc import Iterable
from typing import cast

import torch

from .symbol import Symbol
from .grammar import Variable
from .cnf_cfg import ChomskyNormalFormContextFreeGrammar

class ChomskyNormalFormContextFreeGrammarMembershipTester:

    _start_variable: Variable
    _has_empty_rule: bool
    _binary_rules: dict[tuple[Variable, Variable], list[Variable]]
    _terminal_rules: list[set[Variable]]

    def __init__(self, G: ChomskyNormalFormContextFreeGrammar) -> None:
        super().__init__()
        self._start_variable = G.start_variable()
        self._has_empty_rule = G.has_empty_rule()
        self._binary_rules = {}
        for binary_rule in G.binary_rules():
            self._binary_rules.setdefault(binary_rule.right, []).append(binary_rule.left)
        self._terminal_rules = [set() for _ in G.terminals()]
        for terminal_rule in G.terminal_rules():
            self._terminal_rules[terminal_rule.terminal].add(terminal_rule.left)

    def generates(self, string: Iterable[Symbol]) -> bool:
        items: list[list[set[Variable]]] = []
        j = -1
        for j, symbol in enumerate(string):
            items_j: list[set[Variable]] = [set() for _ in range(j)]
            items_j.append(self._terminal_rules[symbol])
            items.append(items_j)
            for span_size in range(2, j+2):
                i = j - span_size + 1
                for k in range(i, j):
                    first_set = items[k][i]
                    if first_set:
                        second_set = items[j][k+1]
                        if second_set:
                            for first in first_set:
                                for second in second_set:
                                    rules = self._binary_rules.get((first, second))
                                    if rules:
                                        items[j][i].update(rules)
        if j == -1:
            return self._has_empty_rule
        else:
            return self._start_variable in items[j][0]

    def num_terminals(self) -> int:
        return len(self._terminal_rules)

    def has_empty_rule(self) -> bool:
        return self._has_empty_rule

    def start_variable(self) -> Variable:
        return self._start_variable

class PyTorchChomskyNormalFormContextFreeGrammarMembershipTester:

    _start_variable: Variable
    _has_empty_rule: bool
    # _binary_rules : num_variables x num_variables x num_variables
    # where rule A -> BC exists iff _binary_rules[A, B, C] is True.
    _binary_rules: torch.Tensor
    # _terminal_rules : num_terminals x num_variables
    # where rule A -> a exists iff _terminal_rules[a, A] is True.
    _terminal_rules: torch.Tensor

    def __init__(self, G: ChomskyNormalFormContextFreeGrammar, device: torch.device) -> None:
        super().__init__()
        self._start_variable = G.start_variable()
        self._has_empty_rule = G.has_empty_rule()
        num_variables = G.num_variables()
        num_terminals = G.num_terminals()
        self._binary_rules = torch.zeros((num_variables, num_variables, num_variables), dtype=torch.bool, device=device)
        for binary_rule in G.binary_rules():
            self._binary_rules[
                G.zero_indexed(binary_rule.left),
                G.zero_indexed(binary_rule.first),
                G.zero_indexed(binary_rule.second)
            ] = True
        self._terminal_rules = torch.zeros((num_terminals, num_variables), dtype=torch.bool, device=device)
        for terminal_rule in G.terminal_rules():
            self._terminal_rules[
                terminal_rule.terminal,
                G.zero_indexed(terminal_rule.left)
            ] = True

    def set_device(self, device: torch.device) -> None:
        self._binary_rules = self._binary_rules.to(device)
        self._terminal_rules = self._terminal_rules.to(device)

    def _device(self) -> torch.device:
        return self._binary_rules.device

    def generates(self, string: Iterable[Symbol]) -> bool:
        # It's necessary for string to be a list so it can be used as an index
        # for self._terminal_rules.
        if not isinstance(string, list):
            string = list(string)
        n = len(string)
        if n == 0:
            return self._has_empty_rule
        num_variables = self.num_variables()
        device = self._device()
        items = torch.zeros((n, n, num_variables), dtype=torch.bool, device=device)
        i = torch.arange(n, device=device)
        j = torch.arange(n, device=device)
        items[i, j] = self._terminal_rules[string]
        for l in range(2, n+1):
            i = i[:-1]
            j = j[:-1]
            j += 1
            # Let A -> BC denote a binary rule.
            # k : (n-l+1) x (l-1)
            # where dims 0, 1 are the indexes i, k respectively.
            k = torch.arange(n-l+1, device=device)[:, None] + torch.arange(l-1, device=device)[None, :]
            # first_items : (n-l+1) x (l-1) x num_variables
            # where dims 0, 1, 2 are i, k, B respectively.
            first_items = items[i[:, None], k]
            # second_items : (n-l+1) x (l-1) x num_variables
            # where dims 0, 1, 2 are j, k, B respectively.
            second_items = items[k+1, j[:, None]]
            items[i, j] = torch.any(
                # The dims inside here are: i, k, A, B, C
                # This includes a lot of useless terms where k <= i or k >= j
                # for the sake of keeping the tensors rectangular. This is safe
                # because those entries will always be false anyway.
                self._binary_rules[None, None] &
                first_items[:, :, None, :, None] &
                second_items[:, :, None, None, :],
                dim=(1, 3, 4)
            )
        return cast(bool, items[0, n-1, self._start_variable - self.num_terminals()].item())

    def num_terminals(self) -> int:
        return self._terminal_rules.size(0)

    def num_variables(self) -> int:
        return self._binary_rules.size(0)

    def has_empty_rule(self) -> bool:
        return self._has_empty_rule

    def start_variable(self) -> Variable:
        return self._start_variable
