
from typing import List, Dict, Tuple
import networkx as nx

from smolagents import tool

from ..syntax.definitions import Message




@tool
def is_a_valid_partial_order(original_unordered_dict: Dict[str, Message], partial_order: List[Tuple[str, str]]) -> bool:
    """
    This is a tool that verifies if the proposed partial order is valid for the original list. The partial order is valid if it does not create cycles and it only uses valid elements of the original list.
    A valid partial order does not imply that the order is correct. It only implies that the order is not contradictory.
    
    Args:
        original_unordered_dict: The original list of messages. The keys are the item names and the values are the item content.
        partial_order: The proposed partial order. Each tuple in the list corresponds to a pair of non-identical item names that should be ordered. For example, if the list is [(a, b), (b, c)], it means that a < b < c.
    """
    if len(partial_order) == 0:
        return True

    # Check if the messages are from the original list and non-identical, build graph
    graph = nx.DiGraph()
    for u, v in partial_order:
        if u not in original_unordered_dict or v not in original_unordered_dict or u == v:
            return False
        
        graph.add_edge(u, v)
        
    # Check if the partial order is valid
    try:
        nx.topological_sort(graph)
    except nx.NetworkXUnfeasible:
        return False

    return True
