from recognizers.tools.graph_traversal import graph_vertexes
from .grammar import Variable
from .cnf_cfg import ChomskyNormalFormContextFreeGrammar
from .trim_cnf_cfg import productive_variables

def cnf_cfg_is_trim(G: ChomskyNormalFormContextFreeGrammar) -> bool:
    return cnf_cfg_is_reachable(G) and cnf_cfg_is_productive(G)

def cnf_cfg_is_reachable(G: ChomskyNormalFormContextFreeGrammar) -> bool:
    edges: dict[Variable, set[Variable]] = {}
    for rule in G.binary_rules():
        edges.setdefault(rule.left, set()).update(rule.right)
    reachable_variables = graph_vertexes([G.start_variable()], lambda A: edges.get(A, []))
    return len(reachable_variables) == G.num_variables()

def cnf_cfg_is_productive(G: ChomskyNormalFormContextFreeGrammar) -> bool:
    return len(productive_variables(G)) == G.num_variables() - 1
