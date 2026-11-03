from recognizers.tools.graph_traversal import graph_find
from .dfa import DeterministicFiniteAutomaton
from .dfas_are_isomorphic import dfas_are_isomorphic
from .dfa_product_construction import dfa_intersection_unlabeled_graph
from .minimize_dfa import minimize_dfa
from .complete_dfa import make_dfa_complete

def dfas_are_equivalent_symmetric_difference(A: DeterministicFiniteAutomaton, B: DeterministicFiniteAutomaton) -> bool:
    r"""Check whether two DFAs are equivalent by computing their symmetric
    difference.

    Time complexity: :math:`O(|Q_A| |Q_B| + |\delta_A| + |\delta_B|)`.
    """
    return dfa_difference_is_empty(A, B) and dfa_difference_is_empty(B, A)

def dfas_are_equivalent_minimization(A: DeterministicFiniteAutomaton, B: DeterministicFiniteAutomaton) -> bool:
    r"""Check whether two DFAs are equivalent by minimizing them and checking
    whether they are isomorphic.

    Time complexity: :math:`O(|Q_A| \log |Q_A| + |Q_B| \log |Q_B|)`.
    """
    return dfas_are_isomorphic(minimize_dfa(A), minimize_dfa(B))

dfas_are_equivalent = dfas_are_equivalent_minimization

def dfa_difference_is_empty(A: DeterministicFiniteAutomaton, B: DeterministicFiniteAutomaton) -> bool:
    # It's necessary to make B complete in order to simulate the intersection
    # with its *complement*.
    roots, successor_func = dfa_intersection_unlabeled_graph(A, make_dfa_complete(B), ordered=False)
    # Check that we can't reach a pair of states where the state from A accepts
    # and the state from B does not accept.
    return not graph_find(roots, successor_func, lambda pair: A.is_accept_state(pair[0]) and not B.is_accept_state(pair[1]))
