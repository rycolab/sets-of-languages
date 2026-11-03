import itertools
from collections.abc import Collection, Iterable, Mapping
from typing import cast

import more_itertools
from orderedsets import OrderedSet

from .grammar import Variable, Terminal
from .cfg import ContextFreeGrammar, Rule
from .cnf_cfg import (
    ChomskyNormalFormContextFreeGrammarContainer,
    BinaryRule,
    TerminalRule
)

def cfg_to_cnf(G: ContextFreeGrammar) -> ChomskyNormalFormContextFreeGrammarContainer:
    # Add a new start variable to ensure that the start variable isn't on the
    # right side of any rule.
    num_variables = G.num_variables()
    num_terminals = G.num_terminals()
    S = Variable(num_terminals + num_variables)
    num_variables += 1
    rules: Iterable[Rule] = itertools.chain([
        Rule(S, (G.start_variable(),))],
        G.rules()
    )
    # Remove ε rules.
    S_is_empty, rules = cfg_remove_epsilon_rules(S, rules)
    # Remove unit rules.
    rules_by_left = cfg_remove_unit_rules(num_terminals, rules)
    # Replace terminals with variables.
    num_variables, non_terminal_rules, terminal_rules = cfg_replace_terminals(
        num_terminals,
        num_variables,
        S,
        rules_by_left
    )
    # Binarize.
    num_variables, binary_rules = cfg_binarize(
        num_terminals,
        num_variables,
        S,
        non_terminal_rules
    )
    result = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=num_variables,
        num_terminals=num_terminals,
        start_variable=S
    )
    if S_is_empty:
        result.add_empty_rule()
    for left, first, second in binary_rules:
        result.add_binary_rule(left, first, second)
    for left, terminal in terminal_rules:
        result.add_terminal_rule(left, terminal)
    return result

def cfg_remove_epsilon_rules(S: Variable, rules: Iterable[Rule]) -> tuple[bool, OrderedSet[Rule]]:
    # For each rule A -> ε where A is not S, for each occurrence of an A on the
    # right side of a rule, add a rule with it removed. If there's a rule R ->
    # A, add R -> ε unless R -> ε was already removed. Repeat until all ε rules
    # except S -> ε are removed.
    S_is_empty = False
    epsilon_variables: OrderedSet[Variable] = OrderedSet()
    non_epsilon_rules: OrderedSet[Rule] = OrderedSet()
    for rule in rules:
        if rule.right:
            non_epsilon_rules.add(rule)
        elif rule.left == S:
            S_is_empty = True
        else:
            epsilon_variables.add(rule.left)
    removed_epsilon_variables: set[Variable] = set()
    while epsilon_variables:
        A = epsilon_variables.pop()
        removed_epsilon_variables.add(A)
        new_non_epsilon_rules: OrderedSet[Rule] = OrderedSet()
        for rule in non_epsilon_rules:
            for new_right in remove_variable(rule.right, A):
                if new_right:
                    new_non_epsilon_rules.add(Rule(rule.left, new_right))
                elif rule.left == S:
                    S_is_empty = True
                elif rule.left not in removed_epsilon_variables:
                    epsilon_variables.add(rule.left)
        non_epsilon_rules = new_non_epsilon_rules
    return S_is_empty, non_epsilon_rules

def remove_variable(
    right: tuple[Variable | Terminal, ...],
    A: Variable
) -> Iterable[tuple[Variable | Terminal, ...]]:
    for positions in more_itertools.powerset((i for i, X in enumerate(right) if X == A)):
        yield remove_positions(right, positions)

def remove_positions[T](
    x: tuple[T, ...],
    positions: tuple[int, ...]
) -> tuple[T, ...]:
    return tuple(itertools.chain.from_iterable(
        x[i+1:j]
        for i, j in itertools.pairwise(itertools.chain([-1], positions, [len(x)]))
    ))

def cfg_remove_unit_rules(
    num_terminals: int,
    rules: Iterable[Rule]
) -> dict[Variable, OrderedSet[tuple[Variable | Terminal, ...]]]:
    unit_rules: OrderedSet[tuple[Variable, Variable]] = OrderedSet()
    rules_by_left: dict[Variable, OrderedSet[tuple[Variable | Terminal, ...]]] = {}
    for rule in rules:
        if len(rule.right) == 1 and rule.right[0] >= num_terminals:
            unit_rules.add((rule.left, cast(Variable, rule.right[0])))
        rules_by_left.setdefault(rule.left, OrderedSet()).add(rule.right)
    removed_unit_rules: set[tuple[Variable, Variable]] = set()
    while unit_rules:
        A, B = unit_rule = unit_rules.pop()
        rules_by_left[A].remove((B,))
        removed_unit_rules.add(unit_rule)
        for right in rules_by_left.get(B, []):
            if not (
                len(right) == 1 and
                right[0] >= num_terminals and
                (A, right[0]) in removed_unit_rules
            ):
                rules_by_left.setdefault(A, OrderedSet()).add(right)
    return rules_by_left

def cfg_replace_terminals(
    num_terminals: int,
    num_variables: int,
    start_variable: Variable,
    rules_by_left: Mapping[Variable, Collection[tuple[Variable | Terminal, ...]]]
) -> tuple[
    int,
    list[tuple[Variable, tuple[Variable, ...]]],
    list[tuple[Variable, Terminal]]
]:
    terminal_rules = []
    terminal_variable_cache = {}
    for left, right_list in rules_by_left.items():
        if left != start_variable and len(right_list) == 1:
            right, = right_list
            if len(right) == 1:
                terminal, = right
                if terminal < num_terminals:
                    terminal_variable_cache[terminal] = left
    def get_terminal_variable(a: Terminal) -> Variable:
        nonlocal num_variables
        result = terminal_variable_cache.get(a)
        if result is None:
            result = terminal_variable_cache[a] = Variable(num_terminals + num_variables)
            num_variables += 1
            terminal_rules.append((result, a))
        return result
    non_terminal_rules = []
    for left, right_list in rules_by_left.items():
        for right in right_list:
            if len(right) == 1 and right[0] < num_terminals:
                terminal_rules.append((left, cast(Terminal, right[0])))
            else:
                non_terminal_rules.append((left, tuple(
                    (
                        get_terminal_variable(cast(Terminal, X))
                        if X < num_terminals
                        else cast(Variable, X)
                    )
                    for X in right
                )))
    return num_variables, non_terminal_rules, terminal_rules

def cfg_binarize(
    num_terminals: int,
    num_variables: int,
    start_variable: Variable,
    rules: Collection[tuple[Variable, tuple[Variable, ...]]]
) -> tuple[
    int,
    list[tuple[Variable, Variable, Variable]]
]:
    rules_by_left: dict[Variable, list[tuple[Variable, ...]]] = {}
    for left, right in rules:
        if left != start_variable:
            rules_by_left.setdefault(left, []).append(right)
    chain_variable_cache = {}
    for left, right_list in rules_by_left.items():
        if len(right_list) == 1:
            right, = right_list
            chain_variable_cache[right] = left
    new_rules = []
    def get_chain_variable(s: tuple[Variable, ...]) -> Variable:
        nonlocal num_variables
        assert len(s) >= 2
        result = chain_variable_cache.get(s)
        if result is None:
            result = chain_variable_cache[s] = Variable(num_terminals + num_variables)
            num_variables += 1
            C = s[1] if len(s) == 2 else get_chain_variable(s[1:])
            new_rules.append((result, s[0], C))
        return result
    for left, right in rules:
        if len(right) < 2:
            raise ValueError(
                'all rules must have at least 2 variables on the right side'
            )
        elif len(right) == 2:
            new_rules.append((left, *right))
        else:
            new_rules.append((left, right[0], get_chain_variable(right[1:])))
    return num_variables, new_rules
