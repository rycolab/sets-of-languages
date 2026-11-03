from .sfre import (
    StarFreeRegularExpression,
    EmptySetRegularExpression,
    EmptyStringRegularExpression,
    SymbolRegularExpression,
    ConcatenationRegularExpression,
    UnionRegularExpression,
    ComplementRegularExpression
)

def simplify_sfre(R: StarFreeRegularExpression) -> StarFreeRegularExpression:
    r"""Simplify a star-free regular expression using some simple heuristics."""
    match R:
        case EmptySetRegularExpression() | EmptyStringRegularExpression() | SymbolRegularExpression():
            return R
        case ConcatenationRegularExpression(first, second):
            simplified_first = simplify_sfre(first)
            simplified_second = simplify_sfre(second)
            children = [simplified_first, simplified_second]
            # Check for concatenation with empty set.
            if any(isinstance(child, EmptySetRegularExpression) for child in children):
                return EmptySetRegularExpression()
            else:
                # Check for concatenation with empty string.
                empty: list[StarFreeRegularExpression] = []
                non_empty: list[StarFreeRegularExpression] = []
                for child in children:
                    (empty if isinstance(child, EmptyStringRegularExpression) else non_empty).append(child)
                if empty:
                    if non_empty:
                        return non_empty[0]
                    else:
                        return empty[0]
                else:
                    return ConcatenationRegularExpression(simplified_first, simplified_second)
        case UnionRegularExpression(first, second):
            simplified_first = simplify_sfre(first)
            simplified_second = simplify_sfre(second)
            children = [simplified_first, simplified_second]
            # Check for union with empty set.
            empty = []
            non_empty = []
            for child in children:
                (empty if isinstance(child, EmptySetRegularExpression) else non_empty).append(child)
            if empty:
                if non_empty:
                    return non_empty[0]
                else:
                    return empty[0]
            else:
                return UnionRegularExpression(simplified_first, simplified_second)
        case ComplementRegularExpression(argument):
            simplified_argument = simplify_sfre(argument)
            # Check for double complement.
            if isinstance(simplified_argument, ComplementRegularExpression):
                return simplified_argument.argument
            else:
                return ComplementRegularExpression(simplified_argument)
        case _:
            raise TypeError
