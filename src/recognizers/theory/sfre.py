import dataclasses

from .symbol import Symbol

class StarFreeRegularExpression:
    pass

class EmptySetRegularExpression(StarFreeRegularExpression):

    def __str__(self) -> str:
        return '∅'

class EmptyStringRegularExpression(StarFreeRegularExpression):

    def __str__(self) -> str:
        return 'ε'

@dataclasses.dataclass
class SymbolRegularExpression(StarFreeRegularExpression):
    symbol: Symbol

    def __str__(self) -> str:
        return str(self.symbol)

@dataclasses.dataclass
class ConcatenationRegularExpression(StarFreeRegularExpression):
    first: StarFreeRegularExpression
    second: StarFreeRegularExpression

    def __str__(self) -> str:
        return f'({self.first})({self.second})'

@dataclasses.dataclass
class UnionRegularExpression(StarFreeRegularExpression):
    first: StarFreeRegularExpression
    second: StarFreeRegularExpression

    def __str__(self) -> str:
        return f'({self.first})∪({self.second})'

@dataclasses.dataclass
class ComplementRegularExpression(StarFreeRegularExpression):
    argument: StarFreeRegularExpression

    def __str__(self) -> str:
        return f'({self.argument})′'
