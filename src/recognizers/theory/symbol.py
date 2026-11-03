import enum
from typing import NewType

Symbol = NewType('Symbol', int)

class ReservedSymbol(enum.Enum):
    EPSILON = enum.auto()
    EOS = enum.auto()
