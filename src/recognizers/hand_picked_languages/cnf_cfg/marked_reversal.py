from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammarContainer

def marked_reversal_cnf_cfg() -> tuple[ChomskyNormalFormContextFreeGrammarContainer, list[str]]:
    G = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=6,
        num_terminals=3
    )
    S, T, TA, TB, A, B = G.variables()
    a, b, m = G.terminals()
    G.add_rules_from_parts([
        (S, A, TA),
        (S, B, TB),
        (S, m),
        (T, A, TA),
        (T, B, TB),
        (T, m),
        (TA, T, A),
        (TB, T, B),
        (A, a),
        (B, b)
    ])
    return G, ['0', '1', '#']
