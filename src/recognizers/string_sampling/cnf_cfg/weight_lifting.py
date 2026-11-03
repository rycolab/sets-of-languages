import math
from collections.abc import Callable

import torch

from recognizers.theory.cfg import (
    Variable,
    Rule
)
from recognizers.theory.cnf_cfg import (
    ChomskyNormalFormContextFreeGrammar,
    EmptyRule,
    BinaryRule,
    TerminalRule
)
from recognizers.theory.cnf_wcfg import ChomskyNormalFormWeightedContextFreeGrammarContainer
from recognizers.theory.log_binning_semiring import LogBinningSemiring
from recognizers.string_sampling.sampler_util import count_to_bins

def lift_cnf_cfg_weights(
    G: ChomskyNormalFormContextFreeGrammar,
    *,
    max_length: int,
    dtype: torch.dtype,
    device: torch.device
) -> ChomskyNormalFormWeightedContextFreeGrammarContainer[torch.Tensor]:
    r"""Given a CFG in CNF, assign uniform probabilities to make it a PCFG in
    CNF, then lift it to the log binning semiring.
    """
    weight_size = max_length + 1
    num_terminals = G.num_terminals()
    result = ChomskyNormalFormWeightedContextFreeGrammarContainer(
        num_variables=G.num_variables(),
        num_terminals=num_terminals,
        start_variable=G.start_variable(),
        semiring=LogBinningSemiring(weight_size)
    )
    grouped_rules: list[list[EmptyRule | BinaryRule | TerminalRule]] = [[] for A in G.variables()]
    for rule in G.rules():
        grouped_rules[rule.left - num_terminals].append(rule)
    for A_int, rules in enumerate(grouped_rules, G.num_terminals()):
        A = Variable(A_int)
        num_rules = len(rules)
        if num_rules > 0:
            log_prob = -math.log(num_rules)
            for rule in rules:
                match rule:
                    case EmptyRule():
                        result.set_empty_rule_weight(
                            count_to_bins(0, log_prob, weight_size, dtype, device)
                        )
                    case BinaryRule():
                        result.set_binary_rule_weight(
                            rule.left,
                            rule.first,
                            rule.second,
                            count_to_bins(0, log_prob, weight_size, dtype, device)
                        )
                    case TerminalRule():
                        result.set_terminal_rule_weight(
                            rule.left,
                            rule.terminal,
                            count_to_bins(1, log_prob, weight_size, dtype, device)
                        )
                    case _:
                        raise TypeError
    return result
