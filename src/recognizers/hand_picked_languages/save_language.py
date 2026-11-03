import argparse
import re

import torch

from recognizers.hand_picked_languages.dfa.all_strings import all_strings_dfa
from recognizers.hand_picked_languages.dfa.empty_set import empty_set_dfa
from recognizers.hand_picked_languages.dfa.repeat_01 import repeat_01_dfa
from recognizers.hand_picked_languages.dfa.even_pairs import even_pairs_dfa
from recognizers.hand_picked_languages.dfa.modular_arithmetic import modular_arithmetic_dfa
from recognizers.hand_picked_languages.dfa.parity import parity_dfa
from recognizers.hand_picked_languages.dfa.cycle_navigation import cycle_navigation_dfa
from recognizers.hand_picked_languages.dfa.first import first_dfa
from recognizers.hand_picked_languages.dfa.dyck_k_m import dyck_k_m_dfa
from recognizers.hand_picked_languages.cnf_cfg.marked_reversal import marked_reversal_cnf_cfg

def get_language(name):
    match name:
        case 'all-strings':
            return all_strings_dfa()
        case 'empty-set':
            return empty_set_dfa()
        case 'repeat-01':
            return repeat_01_dfa()
        case 'even-pairs':
            return even_pairs_dfa()
        case 'modular-arithmetic':
            return modular_arithmetic_dfa()
        case 'parity':
            return parity_dfa()
        case 'cycle-navigation':
            return cycle_navigation_dfa()
        case 'first':
            return first_dfa()
        case _ if (match := re.match(r'^dyck-(\d+)-(\d+)$', name)):
            k = int(match.group(1))
            m = int(match.group(2))
            return dyck_k_m_dfa(k, m)
        case 'marked-reversal-cfg':
            return marked_reversal_cnf_cfg()
        case _:
            raise ValueError(f'invalid language name: {name}')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    language, alphabet = get_language(args.name)
    data = dict(language=language)
    if alphabet is not None:
        data['alphabet'] = alphabet
    torch.save(data, args.output)

if __name__ == '__main__':
    main()
