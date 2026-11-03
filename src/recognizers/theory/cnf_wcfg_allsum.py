import torch

from .cnf_wcfg import ChomskyNormalFormWeightedContextFreeGrammar
from .fixed_point_iteration import fixed_point_iteration

def cnf_wcfg_partition[T](
    G: ChomskyNormalFormWeightedContextFreeGrammar[T],
    dtype: torch.dtype,
    device: torch.device
) -> T:
    semiring = G.semiring()
    # Precompute the terms of the partition function that don't change.
    constant_partition_weights = semiring.zeros((G.num_variables(),), dtype, device)
    empty_rule_weight = G.empty_rule_weight()
    if empty_rule_weight is not None:
        semiring.add_in_place(
            semiring.transform_tensors(
                constant_partition_weights,
                lambda x: x[G.zero_indexed(G.start_variable())]
            ),
            empty_rule_weight
        )
    for terminal_rule in G.terminal_rules():
        semiring.add_in_place(
            semiring.transform_tensors(
                constant_partition_weights,
                lambda x: x[G.zero_indexed(terminal_rule.left)]
            ),
            terminal_rule.weight
        )

    def step(old_partition_weights: T) -> T:
        # Start by adding all the constant terms (by cloning the tensor of
        # constant weights).
        new_partition_weights = semiring.transform_tensors(
            constant_partition_weights,
            lambda x: x.clone()
        )
        for binary_rule in G.binary_rules():
            semiring.add_in_place(
                semiring.transform_tensors(
                    new_partition_weights,
                    lambda x: x[G.zero_indexed(binary_rule.left)]
                ),
                semiring.multiply(
                    semiring.multiply(
                        binary_rule.weight,
                        semiring.transform_tensors(
                            old_partition_weights,
                            lambda x: x[G.zero_indexed(binary_rule.first)]
                        )
                    ),
                    semiring.transform_tensors(
                        old_partition_weights,
                        lambda x: x[G.zero_indexed(binary_rule.second)]
                    )
                )
            )
        return new_partition_weights

    # Run fixed point iteration to solve the system of quadratic equations.
    return fixed_point_iteration(step, semiring.equal, constant_partition_weights)

def cnf_wcfg_allsum[T](
    G: ChomskyNormalFormWeightedContextFreeGrammar[T],
    dtype: torch.dtype,
    device: torch.device
) -> T:
    semiring = G.semiring()
    S = G.start_variable()
    return semiring.transform_tensors(cnf_wcfg_partition(G, dtype, device), lambda x: x[S])
