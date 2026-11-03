import argparse
import pathlib
from collections.abc import Callable

import graphviz
import torch

from recognizers.theory.symbol import Symbol, ReservedSymbol
from recognizers.theory.automaton import State
from recognizers.theory.nfa import NondeterministicFiniteAutomaton, SymbolOrEpsilon

def draw_nfa(
    M: NondeterministicFiniteAutomaton,
    filename: pathlib.Path | str,
    alphabet: list[str] | None = None
) -> None:
    if not isinstance(filename, pathlib.Path):
        filename = pathlib.Path(filename)
    file_format = filename.suffix.lstrip('.')
    graph = graphviz.Digraph(format=file_format)
    graph.attr(rankdir='LR')
    # Start arrow node.
    start_anchor = 'start'
    graph.node(start_anchor, label='', shape='none', width='0', height='0')
    # Create state nodes.
    for state in M.states():
        graph.node(
            f'state{state}',
            label=str(state),
            shape='doublecircle' if M.is_accept_state(state) else 'circle'
        )
    # Start arrow.
    graph.edge(start_anchor, f'state{M.start_state()}')
    # Combine parallel transitions with the same source and destination.
    edge_symbols: dict[tuple[State, State], list[SymbolOrEpsilon]] = {}
    for transition in M.transitions():
        symbol = transition.symbol
        edge_symbols.setdefault((transition.state_from, transition.state_to), []).append(symbol)
    symbol_to_str: Callable[[Symbol], str]
    if alphabet is not None:
        if len(alphabet) != M.alphabet_size():
            raise ValueError(
                f'the length of the alphabet list ({len(alphabet)}) does not '
                f'match the NFA\'s alphabet size ({M.alphabet_size()})'
            )
        def symbol_to_str(a: Symbol) -> str:
            return alphabet[a]
    else:
        symbol_to_str = str
    def label_to_str(a: SymbolOrEpsilon) -> str:
        return 'ε' if a == ReservedSymbol.EPSILON else symbol_to_str(a)
    for (state_from, state_to), symbols in edge_symbols.items():
        graph.edge(
            f'state{state_from}',
            f'state{state_to}',
            label=', '.join(map(label_to_str, sorted(symbols, key=lambda x: -1 if x == ReservedSymbol.EPSILON else x))),
        )
    bare_name = filename.with_suffix('')
    print(f'writing {bare_name}.{file_format}')
    graph.render(
        filename=bare_name,
        format=file_format,
        cleanup=True
    )

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('language', type=pathlib.Path)
    parser.add_argument('output', type=pathlib.Path)
    args = parser.parse_args()

    M = torch.load(args.language, weights_only=False)['language']
    assert isinstance(M, NondeterministicFiniteAutomaton)
    draw_nfa(M, args.output)

if __name__ == '__main__':
    main()
