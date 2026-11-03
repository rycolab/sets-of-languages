from recognizers.tools.graph_traversal import graph_has_cycle
from .cfg import Variable
from .cnf_cfg import ChomskyNormalFormContextFreeGrammar

def trim_cnf_cfg_is_infinite(G: ChomskyNormalFormContextFreeGrammar) -> bool:
    edges: dict[Variable, set[Variable]] = {}
    for rule in G.binary_rules():
        edges.setdefault(rule.left, set()).update(rule.right)
    return graph_has_cycle([G.start_variable()], lambda A: edges.get(A, []))
