from collections.abc import Callable, Hashable, Iterable
from typing import TypeVar

from orderedsets import OrderedSet

def graph_vertexes[Vertex: Hashable](
    roots: Iterable[Vertex],
    successor_func: Callable[[Vertex], Iterable[Vertex]]
) -> set[Vertex]:
    r"""Return the set of all vertexes that are reachable from the root
    vertexes.

    :param roots: It's *not* necessarily safe to have duplicate roots.
        ``successor_func`` will be called on every duplicate.
    :param successor_func: It's safe to return deplicate destination vertexes;
        they will be deduplicated automatically.
    :return: An unordered set of all reachable vertexes.
    """
    agenda = list(roots)
    discovered = set(agenda)
    while agenda:
        u = agenda.pop()
        for v in successor_func(u):
            if v not in discovered:
                discovered.add(v)
                agenda.append(v)
    return discovered

def graph_vertexes_ordered[Vertex: Hashable](
    roots: Iterable[Vertex],
    successor_func: Callable[[Vertex], Iterable[Vertex]]
) -> OrderedSet[Vertex]:
    r"""Like :py:func:`reachable`, but the vertexes are visited and returned in
    a deterministic order that doesn't depend on the hash seed.

    :param roots: It's safe to have duplicate roots.
    :param successor_func: It's safe to return deplicate destination vertexes;
        they will be deduplicated automatically.
    :return: An ordered set of all reachable vertexes.
    """
    discovered = OrderedSet(roots)
    agenda = list(discovered)
    while agenda:
        u = agenda.pop()
        for v in successor_func(u):
            if v not in discovered:
                discovered.add(v)
                agenda.append(v)
    return discovered

def graph_vertexes_and_edges_ordered[Vertex: Hashable, Label](
    roots: Iterable[Vertex],
    successor_func: Callable[[Vertex], Iterable[tuple[Label, Vertex]]]
) -> tuple[OrderedSet[Vertex], list[tuple[Vertex, Label, Vertex]]]:
    r"""Return all of the vertexes and edges that are reachable from the root
    vertexes in a deterministic order that doesn't depend on the hash seed.

    :param roots: It's safe to have duplicate roots.
    :param successor_func: It's safe to return deplicate destination vertexes;
        they will be deduplicated automatically.
    :return: An ordered set of all reachable vertexes and a list of all edges.
    """
    discovered = OrderedSet(roots)
    edges = []
    agenda = list(discovered)
    while agenda:
        u = agenda.pop()
        for a, v in successor_func(u):
            edges.append((u, a, v))
            if v not in discovered:
                discovered.add(v)
                agenda.append(v)
    return discovered, edges

def graph_find[Vertex: Hashable](
    roots: Iterable[Vertex],
    successor_func: Callable[[Vertex], Iterable[Vertex]],
    predicate: Callable[[Vertex], bool]
) -> bool:
    r"""Return whether there is a reachable vertex that satisfies a certain
    predicate.

    :param roots: It's *not* necessarily safe to have duplicate roots.
        ``successor_func`` will be called on every duplicate.
    :param successor_func: It's safe to return deplicate destination vertexes;
        they will be deduplicated automatically.
    :param predicate: A predicate to be tested on each vertex.
    :return: Whether any reachable vertex satisfies ``predicate``.
    """
    agenda = list(roots)
    discovered = set(agenda)
    while agenda:
        u = agenda.pop()
        if predicate(u):
            return True
        for v in successor_func(u):
            if v not in discovered:
                discovered.add(v)
                agenda.append(v)
    return False

def graph_has_cycle[Vertex: Hashable](
    roots: Iterable[Vertex],
    successor_func: Callable[[Vertex], Iterable[Vertex]]
) -> bool:
    r"""Return whether a graph contains a cycle.

    :param roots: It's safe to have duplicate roots.
    :param successor_func: It's safe to return deplicate destination vertexes;
        they will be deduplicated automatically.
    :return: Whether the graph contains a cycle.
    """
    seen = set()
    finished = set()
    for root in roots:
        if root not in finished:
            seen.add(root)
            stack = [(root, iter(successor_func(root)))]
            while stack:
                u, it = stack[-1]
                try:
                    v = next(it)
                except StopIteration:
                    stack.pop()
                    finished.add(u)
                else:
                    if v not in finished:
                        if v in seen:
                            return True
                        else:
                            seen.add(v)
                            stack.append((v, iter(successor_func(v))))
    return False
