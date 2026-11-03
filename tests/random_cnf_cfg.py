import random

from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammarContainer
from recognizers.tools.random_utils import (
    sample_from_negative_binomial,
    sample_from_bernoulli
)

def random_cnf_cfg(
    mean_num_variables: float,
    mean_num_terminals: float,
    rule_probability: float,
    generator: random.Random
) -> ChomskyNormalFormContextFreeGrammarContainer:
    num_variables = sample_from_negative_binomial(mean_num_variables, 1, generator)
    num_terminals = sample_from_negative_binomial(mean_num_terminals, 1, generator)
    G = ChomskyNormalFormContextFreeGrammarContainer(
        num_variables=num_variables,
        num_terminals=num_terminals
    )
    S = G.start_variable()
    non_start_variables = [A for A in G.variables() if A != S]
    if sample_from_bernoulli(rule_probability, generator):
        G.add_empty_rule()
    for A in G.variables():
        for B in non_start_variables:
            for C in non_start_variables:
                if sample_from_bernoulli(rule_probability, generator):
                    G.add_binary_rule(A, B, C)
        for a in G.terminals():
            if sample_from_bernoulli(rule_probability, generator):
                G.add_terminal_rule(A, a)
    return G
