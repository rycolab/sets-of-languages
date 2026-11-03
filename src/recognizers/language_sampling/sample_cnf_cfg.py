import math
import random

import torch

from recognizers.tools.random_utils import (
    sample_from_negative_binomial,
    sample_from_bernoulli
)
from recognizers.theory.cnf_cfg import (
    ChomskyNormalFormContextFreeGrammarContainer
)
from recognizers.theory.cnf_cfg_generates import (
    PyTorchChomskyNormalFormContextFreeGrammarMembershipTester
)
from recognizers.theory.cnf_cfg_is_infinite import trim_cnf_cfg_is_infinite
from recognizers.theory.trim_cnf_cfg import trim_cnf_cfg
from .density import has_max_density

def sample_cnf_cfg(
    mean_num_variables: float,
    mean_num_terminals: float,
    empty_rule_probability: float,
    mean_num_binary_rules_per_variable: float,
    mean_num_terminal_rules_per_variable: float,
    mean_num_self_embedding_chains: float,
    mean_self_embedding_chain_length: float,
    max_density: float,
    num_density_samples: int,
    density_length_range: tuple[int, int],
    generator: random.Random,
    device: torch.device
) -> ChomskyNormalFormContextFreeGrammarContainer:
    while True:
        num_variables = sample_from_negative_binomial(mean_num_variables, 1, generator)
        num_terminals = sample_from_negative_binomial(mean_num_terminals, 1, generator)
        G = ChomskyNormalFormContextFreeGrammarContainer(
            num_variables=num_variables,
            num_terminals=num_terminals
        )
        S = G.start_variable()
        non_start_variables = [A for A in G.variables() if A != S]
        normalized_binary_rule_probability = min(0.9, mean_num_binary_rules_per_variable / (num_variables-1)**2 if num_variables > 1 else math.inf)
        normalized_terminal_rule_probability = min(0.9, mean_num_terminal_rules_per_variable / num_terminals)
        if sample_from_bernoulli(empty_rule_probability, generator):
            G.add_empty_rule()
        for A in G.variables():
            for B in non_start_variables:
                for C in non_start_variables:
                    if sample_from_bernoulli(normalized_binary_rule_probability, generator):
                        G.add_binary_rule(A, B, C)
            for a in G.terminals():
                if sample_from_bernoulli(normalized_terminal_rule_probability, generator):
                    G.add_terminal_rule(A, a)
        trim_G = trim_cnf_cfg(G)
        is_definitely_infinite = False
        if trim_G.num_variables() > 1:
            num_self_embedding_chains = sample_from_negative_binomial(
                mean_num_self_embedding_chains,
                0,
                generator
            )
            is_definitely_infinite = num_self_embedding_chains > 0
            for _ in range(num_self_embedding_chains):
                add_self_embedding_chain(trim_G, mean_self_embedding_chain_length, generator)
        if is_definitely_infinite or trim_cnf_cfg_is_infinite(trim_G):
            # Check that the density of the language is not too high.
            indexed_G = PyTorchChomskyNormalFormContextFreeGrammarMembershipTester(trim_G, device)
            if has_max_density(
                lambda s: indexed_G.generates(s),
                max_density,
                num_density_samples,
                density_length_range,
                num_terminals,
                generator
            ):
                return trim_G

def add_self_embedding_chain(
    G: ChomskyNormalFormContextFreeGrammarContainer,
    mean_self_embedding_chain_length: float,
    generator: random.Random
) -> None:
    num_rules = sample_from_negative_binomial(mean_self_embedding_chain_length, 2, generator)
    num_left_rules = generator.randint(1, num_rules - 1)
    add_to_left: list[bool] = []
    add_to_left.extend(True for _ in range(num_left_rules))
    add_to_left.extend(False for _ in range(num_rules - num_left_rules))
    generator.shuffle(add_to_left)
    S = G.start_variable()
    non_start_variables = [A for A in G.variables() if A != S]
    A_root = generator.choice(non_start_variables)
    A_prev = A_root
    for i in range(num_rules):
        if i < num_rules - 1:
            A_next = generator.choice(non_start_variables)
        else:
            A_next = A_root
        B = generator.choice(non_start_variables)
        if add_to_left[i]:
            G.add_binary_rule(A_prev, B, A_next)
        else:
            G.add_binary_rule(A_prev, A_next, B)
        A_prev = A_next
