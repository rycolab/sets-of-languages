from recognizers.tools.graph_traversal import graph_vertexes
from .cnf_cfg import (
    ChomskyNormalFormContextFreeGrammar,
    ChomskyNormalFormContextFreeGrammarContainer,
    Variable
)

def trim_cnf_cfg(G: ChomskyNormalFormContextFreeGrammar) -> ChomskyNormalFormContextFreeGrammarContainer:
    # Get the set of variables that generate at least one string.
    new_variables = productive_variables(G)
    # Get the set of non-empty variables reachable from the start variable.
    edges: dict[Variable, set[Variable]] = {}
    for rule in G.binary_rules():
        # There is no need to check if rule.left is in new_variables, because if
        # both rule.first and rule.second are both productive, then it is
        # implied that rule.left is also productive.
        if rule.first in new_variables and rule.second in new_variables:
            edges.setdefault(rule.left, set()).update(rule.right)
    new_variables = graph_vertexes([G.start_variable()], lambda A: edges.get(A, []))
    # We have to reindex the variables. The terminals do not change.
    sorted_new_variables = [A for A in G.variables() if A in new_variables]
    variable_map = { A : Variable(i) for i, A in enumerate(sorted_new_variables, G.num_terminals()) }
    result = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=len(sorted_new_variables),
        num_terminals=G.num_terminals(),
        start_variable=variable_map[G.start_variable()]
    )
    if G.has_empty_rule():
        result.add_empty_rule()
    for binary_rule in G.binary_rules():
        if (
            binary_rule.left in new_variables and
            binary_rule.first in new_variables and
            binary_rule.second in new_variables
        ):
            result.add_binary_rule(
                variable_map[binary_rule.left],
                variable_map[binary_rule.first],
                variable_map[binary_rule.second]
            )
    for terminal_rule in G.terminal_rules():
        if terminal_rule.left in new_variables:
            result.add_terminal_rule(
                variable_map[terminal_rule.left],
                terminal_rule.terminal
            )
    return result

def productive_variables(G: ChomskyNormalFormContextFreeGrammar) -> set[Variable]:
    r"""Get the set of non-start variables that generate at least one string.

    The start variable is omitted because it can never be trimmed anyway.
    """
    S = G.start_variable()
    binary_rules = [rule for rule in G.binary_rules() if rule.left != S]
    preterminals = { rule.left for rule in G.terminal_rules() if rule.left != S }
    curr_variables = preterminals
    while True:
        next_variables = preterminals.copy()
        for rule in binary_rules:
            if rule.first in curr_variables and rule.second in curr_variables:
                next_variables.add(rule.left)
        if len(next_variables) == len(curr_variables):
            return curr_variables
        curr_variables = next_variables
