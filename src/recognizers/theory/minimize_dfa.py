from typing import cast

from orderedsets import OrderedSet

from .automaton import State
from .dfa import DeterministicFiniteAutomaton, DeterministicFiniteAutomatonContainer
from .complete_dfa import CompleteDeterministicFiniteAutomaton
from .nfa import NondeterministicFiniteAutomaton
from .trim_nfa import trim_nfa
from .complete_dfa import make_dfa_complete
from .determinize_nfa import determinize_nfa
from .reverse_nfa import reverse_multi_start_nfa

def minimize_dfa_hopcroft(M: DeterministicFiniteAutomaton) -> DeterministicFiniteAutomatonContainer:
    return minimize_trim_dfa(trim_nfa(M))

def minimize_trim_dfa_hopcroft(M: DeterministicFiniteAutomaton) -> DeterministicFiniteAutomatonContainer:
    min_M = merge_equivalent_states_hopcroft(make_dfa_complete(M))
    remove_trap_state(min_M)
    return min_M

def minimize_nfa_brzozowski(M: NondeterministicFiniteAutomaton) -> DeterministicFiniteAutomaton:
    # Trimming at the beginning guarantees that the output will be trim.
    return minimize_trim_nfa_brzozowski(trim_nfa(M))

def minimize_trim_nfa_brzozowski(M: NondeterministicFiniteAutomaton) -> DeterministicFiniteAutomaton:
    return determinize_nfa(reverse_multi_start_nfa(determinize_nfa(reverse_multi_start_nfa(M))))

minimize_dfa = minimize_dfa_hopcroft
minimize_trim_dfa = minimize_trim_dfa_hopcroft

def merge_equivalent_states_hopcroft(M: CompleteDeterministicFiniteAutomaton) -> DeterministicFiniteAutomatonContainer:
    # We assume that M is a complete, trim DFA.
    # Index all the transitions in reverse.
    backward_transitions: list[list[list[State]]] = [[[] for a in M.alphabet()] for r in M.states()]
    for t in M.transitions():
        backward_transitions[t.state_to][t.symbol].append(t.state_from)
    # Partition the states into non-accept and accept states.
    # Use OrderedSets to ensure consistent iteration order and results.
    partition: list[OrderedSet[State]] = [OrderedSet(), OrderedSet()]
    for q in M.states():
        partition[int(M.is_accept_state(q))].add(q)
    # Filter out any empty blocks. The only way empty blocks can occur is if all
    # states are accepting or non-accepting.
    partition = [block for block in partition if block]
    state_to_block_id = [-1] * M.num_states()
    for block_id, block in enumerate(partition):
        for q in block:
            state_to_block_id[q] = block_id
    # Initialize the agenda with the smaller block.
    agenda = [min(partition, key=len)]
    while agenda:
        splitter = agenda.pop()
        for a in M.alphabet():
            intersections: dict[int, OrderedSet[State]] = {}
            for r in splitter:
                for q in backward_transitions[r][a]:
                    intersections.setdefault(state_to_block_id[q], OrderedSet()).add(q)
            for block_id, intersection in intersections.items():
                block = partition[block_id]
                if len(intersection) < len(block):
                    difference = block - intersection
                    partition[block_id] = intersection
                    new_block_id = len(partition)
                    for q in difference:
                        state_to_block_id[q] = new_block_id
                    partition.append(difference)
                    agenda.append(min(intersection, difference, key=len))
    # Turn the blocks into the states of a DFA.
    result = DeterministicFiniteAutomatonContainer(
        num_states=len(partition),
        alphabet_size=M.alphabet_size(),
        start_state=State(state_to_block_id[M.start_state()])
    )
    for block_id, block in enumerate(partition):
        for a in M.alphabet():
            result.add_transition(State(block_id), a, State(state_to_block_id[M.transition(next(iter(block)), a)]))
        if any(M.is_accept_state(q) for q in block):
            result.add_accept_state(State(block_id))
    return result

def remove_trap_state(M: DeterministicFiniteAutomatonContainer) -> None:
    # We could implement this using trim_nfa(), but we can use a check that's
    # more lightweight than a full DFS.
    # Assume the input is a minimal complete DFA. If there is a trap state, it
    # will be a non-accept state that has self-loops on every symbol.
    for q in M.states():
        if not M.is_accept_state(q) and all(M.transition(q, a) == q for a in M.alphabet()):
            if q != M.start_state():
                M.remove_state(q)
            else:
                # If the trap state is the start state (i.e., this DFA
                # recognizes the empty set), we can't remove the state, but we
                # should remove all of its transitions.
                for a in M.alphabet():
                    M.remove_transition(q, a)
            return
