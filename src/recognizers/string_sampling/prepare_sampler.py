import argparse
from typing import Any

import torch

from recognizers.theory.dfa import DeterministicFiniteAutomaton
from recognizers.string_sampling.dfa.weight_lifting import lift_dfa_weights
from recognizers.string_sampling.dfa.weight_pushing import WeightedDeterministicFiniteAutomatonSampler
from recognizers.theory.cnf_cfg import ChomskyNormalFormContextFreeGrammar
from recognizers.string_sampling.cnf_cfg.weight_lifting import lift_cnf_cfg_weights
from recognizers.string_sampling.cnf_cfg.weight_pushing import ChomskyNormalFormWeightedContextFreeGrammarSampler

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-length', type=int, required=True)
    parser.add_argument('--dtype', choices=['float16', 'float32'], default='float16')
    parser.add_argument('--device', type=torch.device, required=True)
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    device = args.device

    data = torch.load(args.input, weights_only=False)
    language = data.pop('language')
    sampler: Any
    match language:
        case DeterministicFiniteAutomaton():
            dfa = language
            wdfa = lift_dfa_weights(
                dfa,
                max_length=args.max_length,
                dtype=dtype,
                device=device
            )
            sampler = WeightedDeterministicFiniteAutomatonSampler(
                dfa,
                wdfa,
                dtype=dtype,
                device=device
            )
        case ChomskyNormalFormContextFreeGrammar():
            cnf_cfg = language
            # Every time the CFG generates a terminal, increment the count. All
            # terminal rules generate 1 terminal. Other rules generate no
            # terminals.
            cnf_wcfg = lift_cnf_cfg_weights(
                cnf_cfg,
                max_length=args.max_length,
                dtype=dtype,
                device=device
            )
            sampler = ChomskyNormalFormWeightedContextFreeGrammarSampler(
                cnf_cfg,
                cnf_wcfg,
                dtype=dtype,
                device=device
            )
        case _:
            raise ValueError
    data['sampler'] = sampler
    torch.save(data, args.output)

if __name__ == '__main__':
    main()
