import dataclasses
from collections.abc import Iterable, Iterator
from typing import cast

@dataclasses.dataclass
class LinkedListNode[T]:
    value: T
    next: LinkedListNode[T] | None = None

class LinkedList[T]:

    head: LinkedListNode[T] | None
    tail: LinkedListNode[T] | None

    def __init__(self, values: Iterable[T] = ()) -> None:
        self.head = None
        self.tail = None
        for value in values:
            self.append(value)

    def append(self, value: T) -> None:
        self.append_node(LinkedListNode(value))

    def append_node(self, node: LinkedListNode[T]) -> None:
        if self.head is None:
            self.head = self.tail = node
        else:
            cast(LinkedListNode[T], self.tail).next = node
            self.tail = node

    def extend(self, linked_list: LinkedList[T]) -> None:
        # Constant time. The argument linked_list will not be modified
        # immediately, but it can be later if this list is modified afterwards.
        if self.head is None:
            self.head = linked_list.head
        else:
            cast(LinkedListNode[T], self.tail).next = linked_list.head
        self.tail = linked_list.tail

    def values(self) -> Iterable[T]:
        curr = self.head
        while curr is not None:
            yield curr.value
            curr = curr.next

    def __iter__(self) -> Iterator[T]:
        return iter(self.values())
