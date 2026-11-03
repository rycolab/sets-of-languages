from recognizers.tools.graph_traversal import graph_has_cycle

def test_has_cycle_empty() -> None:
    assert not graph_has_cycle([], lambda v: [])

def test_has_cycle_one_vertex() -> None:
    G: dict[int, list[int]] = {0 : []}
    assert not graph_has_cycle([0], lambda v: G[v])

def test_has_cycle_tree() -> None:
    G = {
        0 : [1, 2],
        1 : [3],
        2 : [4, 5],
        3 : [],
        4 : [],
        5 : []
    }
    assert not graph_has_cycle([0], lambda v: G[v])

def test_has_cycle_diamond() -> None:
    G = {
        0 : [1, 2],
        1 : [3],
        2 : [3],
        3 : []
    }
    assert not graph_has_cycle([0], lambda v: G[v])

def test_has_cycle_three_cycle() -> None:
    G = {
        0 : [1],
        1 : [2],
        2 : [0]
    }
    assert graph_has_cycle([0], lambda v: G[v])

def test_has_cycle_diamond_then_cycle() -> None:
    G = {
        0 : [1, 2],
        1 : [3],
        2 : [3],
        3 : [4],
        4 : [5],
        5 : [7, 6],
        6 : [4],
        7 : [8],
        8 : []
    }
    assert graph_has_cycle([0], lambda v: G[v])

def test_has_cycle_multiple_roots() -> None:
    G = {
        1 : [3, 4],
        3 : [5],
        4 : [5],
        5 : [6],
        6 : [],
        2 : [7],
        7 : [8],
        8 : [4, 9],
        9 : [10],
        10 : [8]
    }
    assert graph_has_cycle([1, 2], lambda v: G[v])
    G[10] = []
    assert not graph_has_cycle([1, 2], lambda v: G[v])

def test_has_cycle_at_end_of_very_long_chain() -> None:
    chain_len = 1000
    cycle_len = 5
    def successor_func(v):
        if v < chain_len:
            return [v+1]
        elif v < chain_len + cycle_len:
            return [chain_len + (v - chain_len + 1) % cycle_len]
        else:
            raise ValueError
    assert graph_has_cycle([0], successor_func)
