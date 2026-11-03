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
from recognizers.tools.random_utils import sample_from_negative_binomial

class ExpressionType(enum.Enum):
    EMPTY_SET = enum.auto()
    EMPTY_STRING = enum.auto()
    SYMBOL = enum.auto()
    CONCATENATION = enum.auto()
    UNION = enum.auto()
    COMPLEMENT = enum.auto()

EXPRESSION_TYPES = list(ExpressionType)
ARGUMENT_COUNTS = numpy.array([0, 0, 0, 2, 2, 1])

def random_sfre(
    mean_alphabet_size: float,
    empty_set_weight: float,
    empty_string_weight: float,
    symbol_weight: float,
    concatenation_weight: float,
    union_weight: float,
    complement_weight: float,
    sharpness: float,
    generator: random.Random
) -> tuple[StarFreeRegularExpression, int]:
    weights = numpy.array([
        empty_set_weight,
        empty_string_weight,
        symbol_weight,
        concatenation_weight,
        union_weight,
        complement_weight
    ])
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
    return sample_expression(0), alphabet_size
