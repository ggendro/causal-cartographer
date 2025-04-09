
import networkx as nx
from typing import List, Optional, Set, Generator
from collections import deque
import itertools

from ..syntax.definitions import Message


def update_graph_node_values_in_place(causal_graph: nx.DiGraph, updates: Message, add_causal_effect: bool = True, reset_old_values: bool = False) -> None:
    """
    Update node attributes within a given causal graph based on provided updates.
    This function modifies the attributes of a node in the causal_graph in place. It uses the node name provided
    in the updates dictionary to identify which node to update. Optionally, it can reset all old attribute values 
    (except for "name") and add a causal effect attribute if certain conditions are met.
    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing the causal model, where each node's attributes may be updated.
        updates (Message): A dictionary-like object containing updates to apply to the node. It must include a "name" key specifying 
                           the target node.
        add_causal_effect (bool, optional): If True (default), and if "current_value" is provided in updates but "causal_effect" is not,
                                            sets the node's "causal_effect" attribute to the value of "current_value".
        reset_old_values (bool, optional): If True, clears all existing attributes of the node (except "name") before applying new updates.
    Returns:
        None
    Side Effects:
        Modifies the causal_graph in place by updating the specified node's attributes.
    """
    node_name = updates["name"]

    if reset_old_values:
        for key in list(causal_graph.nodes[node_name].keys()):
            if key != "name":
                del causal_graph.nodes[node_name][key]

    for key, value in updates.items():
        if key != "name":
            causal_graph.nodes[node_name][key] = value

    if add_causal_effect and "current_value" in updates and "causal_effect" not in updates:
        causal_graph.nodes[node_name]["causal_effect"] = updates["current_value"]


def build_intervened_graph(causal_graph: nx.DiGraph, interventions: List[Message], structure_only: bool = False) -> nx.DiGraph:
    """
    Build a new intervened causal graph based on the provided interventions.

    This function creates a copy of the given causal graph and applies modifications
    for each intervention in the list. For each intervention, if the structure_only flag is
    False, the function updates the node values using the update_graph_node_values_in_place function.
    Regardless of the flag, it removes all incoming edges to the node specified in the intervention,
    thereby "cutting off" the node's influences in the graph.

    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing the original causal structure.
        interventions (List[Message]): A list of intervention messages, each containing at least a "name"
            key that denotes the target node for the intervention.
        structure_only (bool): If True, only the graph's structure is modified (i.e., by removing incoming
            edges) and node values remain unchanged. Defaults to False.

    Returns:
        nx.DiGraph: A new causal graph that reflects the interventions applied.
    """
    intervened_graph = causal_graph.copy()
    for intervention in interventions:
        if not structure_only:
            update_graph_node_values_in_place(intervened_graph, intervention)
        intervened_graph.remove_edges_from(list(intervened_graph.in_edges(intervention["name"])))
    return intervened_graph


def convert_path_to_chain(path: List[str], graph: nx.DiGraph) -> nx.DiGraph:
    """
    Convert a path of nodes into a chain subgraph based on the supplied graph.
    This function creates a directed chain that contains the nodes listed in 
    the input 'path' based on the edges present in the input 'graph'.
    Parameters:
        path (List[str]): A list of node identifiers defining the sequence for the chain.
        graph (nx.DiGraph): A directed graph from which the valid edges are verified.
    Returns:
        nx.DiGraph: A directed subgraph representing the chain with nodes from the input path 
                    and the corresponding valid edges.
    Raises:
        ValueError: If no valid edge exists between a pair of consecutive nodes in the path.
    """
    chain = nx.DiGraph()
    chain.add_nodes_from(path)

    for a, b in zip(path[:-1], path[1:]):
        if graph.has_edge(a, b):
            chain.add_edge(a, b)
        elif graph.has_edge(b, a):
            chain.add_edge(b, a)
        else:
            raise ValueError("Path does not exist in graph")
    
    return chain


def find_causal_paths(causal_graph: nx.DiGraph, source: str, target: str, conditioning_set: List[str], traversal_cutoff: Optional[int] = None) -> List[List[str]]:
    """
    Find all causal paths from the source to the target in a causal graph that are not d-separated by a given conditioning set.
    This function searches for all simple paths between a source and a target node within an undirected version of a causal graph (provided as a directed graph) and then filters these paths based on d-separation. A path is considered valid if the nodes in the provided conditioning set do not d-separate the source and target according to the chain representation derived from the path.
    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing the causal structure.
        source (str): The starting node of the causal path.
        target (str): The ending node of the causal path.
        conditioning_set (List[str]): A list of nodes to be used for checking d-separation.
        traversal_cutoff (Optional[int]): An optional cutoff limiting the maximum number of nodes in a path (depth of traversal). 
                                          If None, no cutoff is applied.
    Returns:
        List[List[str]]: A list of valid causal paths, where each path is represented as a list of node identifiers.
    """
    candidate_paths = nx.all_simple_paths(causal_graph.to_undirected(as_view=True), source=source, target=target, cutoff=traversal_cutoff)

    causal_paths = []
    for path in candidate_paths:
        chain = convert_path_to_chain(path, causal_graph)
        conditioning_subset = set(conditioning_set) & set(chain.nodes)
        if not nx.is_d_separator(chain, source, target, conditioning_subset):
            causal_paths.append(path)

    return causal_paths


def build_target_node_causal_blanket(graph: nx.DiGraph, target_node: str, conditioning_set: List[str], traversal_cutoff: Optional[int] = None) -> Set[str]:
    """
    Construct the causal blanket for the target node by traversing its parent nodes in the graph.
    This function performs a breadth-first search starting from the target node and moves
    upwards through its ancestors (predecessors). It adds each discovered node to the
    causal blanket as long as the node is not blocked by the provided conditioning set.
    Parameters:
        graph (nx.DiGraph): The directed acyclic graph representing causal relationships.
        target_node (str): The node for which the causal blanket is to be constructed.
        conditioning_set (List[str]): A list of observed nodes that block the traversal.
        traversal_cutoff (Optional[int]): An optional maximum depth for the traversal. If the
            traversal exceeds this depth, an nx.ExceededMaxIterations is raised.
    Returns:
        Set[str]: A set containing nodes that form the causal blanket of the target node, i.e. the set of nodes to be inferred from the observations in order to causally infer the target node.
    Raises:
        nx.ExceededMaxIterations: If the current layer during traversal reaches the specified
            traversal_cutoff.
    """
    queue = deque([(target_node, 0)])
    visited = set()

    while queue:
        node, layer = queue.popleft()

        if traversal_cutoff is not None and layer >= traversal_cutoff:
            raise nx.ExceededMaxIterations("Traversal cutoff exceeded")

        if node not in visited:
            visited.add(node)

            if node not in conditioning_set:
                for parent in graph.predecessors(node):
                    if parent not in visited:
                        queue.append((parent, layer + 1))

    return visited


def _dagify_recursive(graph: nx.DiGraph, visited: Set[str], cycles: Generator[List[str], None, None]) -> Generator[nx.DiGraph, None, None]:
    if graph in visited:
        return None
    
    visited.add(graph)
    if nx.is_directed_acyclic_graph(graph):
        yield graph
    else:
        try:
            cycle = next(cycles)
        except StopIteration:
            return None
        
        buffer = cycles
        for edge in zip(cycle, cycle[1:] + cycle[:1]):
            if edge in graph.edges:
                new_graph = graph.copy()
                new_graph.remove_edge(*edge)
                remaining_cycles, buffer = itertools.tee(buffer)
                yield from _dagify_recursive(new_graph, visited, remaining_cycles)

def dagify(graph: nx.DiGraph) -> Generator[nx.DiGraph, None, None]:
    """
    Converts the input directed graph into one or more directed acyclic graphs (DAGs)
    by recursively resolving cycles.

    Parameters:
        graph (nx.DiGraph): A directed graph that may contain cycles.

    Yields:
        nx.DiGraph: A version of the input graph where cycles have been removed, ensuring a DAG.

    Note:
        This function utilizes a helper function '_dagify_recursive' to perform the recursive
        cycle resolution. It first identifies all simple cycles using NetworkX's simple_cycles.
    """
    visited = set()
    cycles = nx.simple_cycles(graph)
    yield from _dagify_recursive(graph, visited, cycles)




__all__ = ['update_graph_node_values_in_place', 'build_intervened_graph', 'convert_path_to_chain', 'find_causal_paths', 'build_target_node_causal_blanket', 'dagify']