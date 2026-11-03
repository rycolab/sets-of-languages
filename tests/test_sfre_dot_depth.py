from recognizers.theory.sfre_dot_depth import sfre_dot_depth
from recognizers.theory.symbol import Symbol
from recognizers.theory.sfre import (
    EmptySetRegularExpression,
    EmptyStringRegularExpression,
    SymbolRegularExpression,
    ConcatenationRegularExpression as cat,
    UnionRegularExpression as union,
    ComplementRegularExpression as comp
)

eset = EmptySetRegularExpression()
eps = EmptyStringRegularExpression()

def sym(a: int) -> SymbolRegularExpression:
    return SymbolRegularExpression(Symbol(a))

def test_examples() -> None:
    a = sym(0)
    assert sfre_dot_depth(a) == 0
    assert sfre_dot_depth(eset) == 0
    assert sfre_dot_depth(eps) == 0
    assert sfre_dot_depth(cat(a, a)) == 0
    assert sfre_dot_depth(cat(cat(a, a), a)) == 0
    assert sfre_dot_depth(union(cat(cat(a, a), a), a)) == 0
    assert sfre_dot_depth(cat(a, union(cat(cat(a, a), a), a))) == 1
    assert sfre_dot_depth(comp(cat(a, union(cat(cat(a, a), a), a)))) == 1
    assert sfre_dot_depth(cat(comp(cat(a, union(cat(cat(a, a), a), a))), eps)) == 2
    assert sfre_dot_depth(comp(union(cat(comp(cat(a, union(cat(cat(a, a), a), a))), eps), a))) == 2
    assert sfre_dot_depth(cat(a, comp(union(cat(comp(cat(a, union(cat(cat(a, a), a), a))), eps), a)))) == 3
