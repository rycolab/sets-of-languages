import enum
import random

import numpy

from recognizers.theory.symbol import Symbol
from recognizers.theory.sfre import (
    StarFreeRegularExpression,
    EmptySetRegularExpression,
    EmptyStringRegularExpression,
    SymbolRegularExpression,
    ConcatenationRegularExpression,
    UnionRegularExpression,
    ComplementRegularExpression
)
from recognizers.theory.simplify_sfre import simplify_sfre
from recognizers.theory.sfre_to_min_dfa import sfre_to_min_dfa
from recognizers.theory.dfa import DeterministicFiniteAutomaton
from recognizers.theory.is_podfa import is_podfa
from recognizers.theory.dfa_accepts import dfa_accepts
from recognizers.tools.random_utils import sample_from_negative_binomial
from .density import has_max_density

class ExpressionType(enum.Enum):
    EMPTY_SET = enum.auto()
    EMPTY_STRING = enum.auto()
    SYMBOL = enum.auto()
    CONCATENATION = enum.auto()
    UNION = enum.auto()
    COMPLEMENT = enum.auto()

EXPRESSION_TYPES = list(ExpressionType)
ARGUMENT_COUNTS = numpy.array([0, 0, 0, 2, 2, 1])

def sample_sfre(
    mean_alphabet_size: float,
    empty_set_weight: float,
    empty_string_weight: float,
    symbol_weight: float,
    concatenation_weight: float,
    union_weight: float,
    complement_weight: float,
    sharpness: float,
    max_density: float,
    num_density_samples: int,
    density_length_range: tuple[int, int],
    generator: random.Random
) -> tuple[StarFreeRegularExpression, DeterministicFiniteAutomaton]:
    weights = numpy.array([
        empty_set_weight,
        empty_string_weight,
        symbol_weight,
        concatenation_weight,
        union_weight,
        complement_weight
    ])
    while True:
        alphabet_size = sample_from_negative_binomial(mean_alphabet_size, 1, generator)
        def sample_expression(depth: int) -> StarFreeRegularExpression:
            expression_type, = generator.choices(
                EXPRESSION_TYPES,
                cum_weights=numpy.cumsum(weights + (sharpness * depth) / (ARGUMENT_COUNTS + 1))
            )
            match expression_type:
                case ExpressionType.EMPTY_SET:
                    return EmptySetRegularExpression()
                case ExpressionType.EMPTY_STRING:
                    return EmptyStringRegularExpression()
                case ExpressionType.SYMBOL:
                    symbol = Symbol(generator.randrange(alphabet_size))
                    return SymbolRegularExpression(symbol)
                case ExpressionType.CONCATENATION:
                    return ConcatenationRegularExpression(
                        sample_expression(depth + 1),
                        sample_expression(depth + 1)
                    )
                case ExpressionType.UNION:
                    return UnionRegularExpression(
                        sample_expression(depth + 1),
                        sample_expression(depth + 1)
                    )
                case ExpressionType.COMPLEMENT:
                    return ComplementRegularExpression(
                        sample_expression(depth + 1)
                    )
        R = simplify_sfre(sample_expression(0))
        # Convert the regular expression to a minimal DFA.
        min_M = sfre_to_min_dfa(R, alphabet_size)
        # Check if the minimal DFA is a PODFA. This will eliminate PODFA
        # languages, including trivial (accepts or rejects all strings) and
        # finite languages.
        # Also check that the density of the language is not too high.
        if (
            not is_podfa(min_M) and
            has_max_density(
                lambda s: dfa_accepts(min_M, s),
                max_density,
                num_density_samples,
                density_length_range,
                alphabet_size,
                generator
            )
        ):
            return R, min_M
