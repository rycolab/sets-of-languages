from .sfre import (
    StarFreeRegularExpression,
    EmptySetRegularExpression,
    EmptyStringRegularExpression,
    SymbolRegularExpression,
    ConcatenationRegularExpression,
    UnionRegularExpression,
    ComplementRegularExpression
)

def sfre_dot_depth(R: StarFreeRegularExpression) -> int:
    dot_depth, saw_boolean = _sfre_dot_depth(R)
    return dot_depth

def _sfre_dot_depth(R: StarFreeRegularExpression) -> tuple[int, bool]:
    match R:
        case EmptySetRegularExpression() | EmptyStringRegularExpression() | SymbolRegularExpression():
            return 0, False
        case ConcatenationRegularExpression(first, second):
            first_dot_depth, first_saw_boolean = _sfre_dot_depth(first)
            second_dot_depth, second_saw_boolean = _sfre_dot_depth(second)
            return (
                max(
                    first_dot_depth + int(first_saw_boolean),
                    second_dot_depth + int(second_saw_boolean)
                ),
                False
            )
        case UnionRegularExpression(first, second):
            first_dot_depth, _ = _sfre_dot_depth(first)
            second_dot_depth, _ = _sfre_dot_depth(second)
            return max(first_dot_depth, second_dot_depth), True
        case ComplementRegularExpression(argument):
            dot_depth, _ = _sfre_dot_depth(argument)
            return dot_depth, True
        case _:
            raise TypeError
