from typing import cast

from .grammar import Variable, Terminal
from .cfg import ContextFreeGrammar

LETTER_S_INDEX = ord('S') - ord('A')

def cfg_to_text(G: ContextFreeGrammar) -> str:
    num_variables = G.num_variables()
    num_terminals = G.num_terminals()
    S = G.start_variable()
    def variable_to_text(A: Variable) -> str:
        if A == S:
            return 'S'
        else:
            index = G.zero_indexed(A)
            num_primes, letter_no = divmod(index - int(A >= S and index <= LETTER_S_INDEX), 26)
            return chr(ord('A') + letter_no) + '′' * num_primes
    def terminal_to_text(a: Terminal) -> str:
        num_primes, letter_no = divmod(a, 26)
        return chr(ord('a') + letter_no) + '′' * num_primes
    def symbol_to_text(X: Variable | Terminal) -> str:
        if G.is_variable(X):
            return variable_to_text(cast(Variable, X))
        else:
            return terminal_to_text(cast(Terminal, X))
    def right_to_text(right: tuple[Variable | Terminal, ...]) -> str:
        return ''.join(map(symbol_to_text, right)) if right else 'ε'
    def rules_to_text(left: Variable, right_list: list[tuple[Variable | Terminal, ...]]) -> str:
        return f'{variable_to_text(left)} → {' | '.join(map(right_to_text, right_list))}'
    grouped_rules: dict[Variable, list[tuple[Variable | Terminal, ...]]] = {}
    for rule in G.rules():
        grouped_rules.setdefault(rule.left, []).append(rule.right)
    return '\n'.join(rules_to_text(left, right_list) for left, right_list in grouped_rules.items())
