from collections.abc import Callable, Hashable, Iterable, MutableSet, Set
from typing import cast

from orderedsets import OrderedSet

from recognizers.tools.graph_traversal import graph_vertexes_and_edges_ordered
from .automaton import State
from .symbol import Symbol
from .dfa import DeterministicFiniteAutomaton, DeterministicFiniteAutomatonContainer

def dfa_union_graph(
    A: DeterministicFiniteAutomaton,
    B: DeterministicFiniteAutomaton,
    ordered: bool = True
) -> tuple[
    list[tuple[State | None, State | None]],
    Callable[
        [tuple[State | None, State | None]],
        Iterable[tuple[Symbol, tuple[State | None, State | None]]]
    ]
]:
    roots: list[tuple[State| None, State | None]] = [(A.start_state(), B.start_state())]
    A_get_symbols = _index_symbols_allow_none(A, ordered)
    B_get_symbols = _index_symbols_allow_none(B, ordered)
    def successor_func(pair: tuple[State | None, State | None]) -> Iterable[tuple[Symbol, tuple[State | None, State | None]]]:
        qa, qb = pair
        return (
            (a, (
                A.transition(qa, a) if qa is not None else None,
                B.transition(qb, a) if qb is not None else None
            ))
            for a in A_get_symbols(qa) | B_get_symbols(qb)
        )
    return roots, successor_func

def dfa_union(
    A: DeterministicFiniteAutomaton,
    B: DeterministicFiniteAutomaton
) -> DeterministicFiniteAutomatonContainer:
    return _graph_to_dfa(
        dfa_union_graph(A, B, ordered=True),
        lambda qa, qb: (
            (qa is not None and A.is_accept_state(qa)) or
            (qb is not None and B.is_accept_state(qb))
        ),
        max(A.alphabet_size(), B.alphabet_size())
    )

def dfa_intersection_unlabeled_graph(
    A: DeterministicFiniteAutomaton,
    B: DeterministicFiniteAutomaton,
    ordered: bool = True
) -> tuple[
    list[tuple[State, State]],
    Callable[[tuple[State, State]], Iterable[tuple[State, State]]]
]:
    roots = [(A.start_state(), B.start_state())]
    A_get_symbols = _index_symbols(A, ordered)
    B_get_symbols = _index_symbols(B, ordered)
    def successor_func(pair: tuple[State, State]) -> Iterable[tuple[State, State]]:
        qa, qb = pair
        return (
            (cast(State, A.transition(qa, a)), cast(State, B.transition(qb, a)))
            for a in A_get_symbols(qa) & B_get_symbols(qb)
        )
    return roots, successor_func

def dfa_intersection_graph(
    A: DeterministicFiniteAutomaton,
    B: DeterministicFiniteAutomaton,
    ordered: bool = True
) -> tuple[
    list[tuple[State, State]],
    Callable[[tuple[State, State]], Iterable[tuple[Symbol, tuple[State, State]]]]
]:
    roots = [(A.start_state(), B.start_state())]
    A_get_symbols = _index_symbols(A, ordered)
    B_get_symbols = _index_symbols(B, ordered)
    def successor_func(pair: tuple[State, State]) -> Iterable[tuple[Symbol, tuple[State, State]]]:
        qa, qb = pair
        return (
            (a, (cast(State, A.transition(qa, a)), cast(State, B.transition(qb, a))))
            for a in A_get_symbols(qa) & B_get_symbols(qb)
        )
    return roots, successor_func

def dfa_intersection(
    A: DeterministicFiniteAutomaton,
    B: DeterministicFiniteAutomaton
) -> DeterministicFiniteAutomatonContainer:
    return _graph_to_dfa(
        dfa_intersection_graph(A, B, ordered=True),
        lambda qa, qb: A.is_accept_state(qa) and B.is_accept_state(qb),
        max(A.alphabet_size(), B.alphabet_size())
    )

def _graph_to_dfa[T: Hashable](
    graph: tuple[list[tuple[T, T]], Callable[[tuple[T, T]], Iterable[tuple[Symbol, tuple[T, T]]]]],
    is_accept: Callable[[T, T], bool],
    alphabet_size: int
) -> DeterministicFiniteAutomatonContainer:
    roots, successor_func = graph
    vertexes, edges = graph_vertexes_and_edges_ordered(roots, successor_func)
    vertex_to_state = { u : State(i) for i, u in enumerate(vertexes) }
    M = DeterministicFiniteAutomatonContainer(
        num_states=len(vertexes),
        alphabet_size=alphabet_size,
        start_state=vertex_to_state[roots[0]]
    )
    for vertex_from, a, vertex_to in edges:
        M.add_transition(vertex_to_state[vertex_from], a, vertex_to_state[vertex_to])
    for (qa, qb), q in vertex_to_state.items():
        if is_accept(qa, qb):
            M.add_accept_state(q)
    return M

def _index_symbols(
    M: DeterministicFiniteAutomaton,
    ordered: bool
) -> Callable[[State], Set[Symbol]]:
    set_class = OrderedSet if ordered else set
    if M.is_complete():
        all_symbols = cast(MutableSet[Symbol], set_class(M.alphabet()))
        def get_symbols(q: State) -> Set[Symbol]:
            return all_symbols
    else:
        index: dict[State, MutableSet[Symbol]] = {}
        for t in M.transitions():
            index.setdefault(t.state_from, cast(MutableSet[Symbol], set_class())).add(t.symbol)
        def get_symbols(q: State) -> Set[Symbol]:
            return index.get(q, cast(MutableSet[Symbol], set_class()))
    return get_symbols

def _index_symbols_allow_none(
    M: DeterministicFiniteAutomaton,
    ordered: bool
) -> Callable[[State | None], Set[Symbol]]:
    get_symbols_no_none = _index_symbols(M, ordered)
    set_class = OrderedSet if ordered else set
    def get_symbols(q: State | None) -> Set[Symbol]:
        return get_symbols_no_none(q) if q is not None else cast(Set[Symbol], set_class())
    return get_symbols
