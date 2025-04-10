
from typing import List, Dict, Tuple
import networkx as nx
import io
import math

from smolagents import tool

from ..syntax.definitions import Message



@tool
def kolmogorov_complexity(inference_graph: nx.DiGraph) -> int:
    """
    This is a tool that computes the Kolmogorov complexity of an inference graph.
    The Kolmogorov complexity of a graph is the length of the shortest program that can generate the graph.
    In this case, the Kolmogorov complexity is the number of nodes and edges in the graph and their attributes, in bits.

    Args:
        inference_graph: A directed graph representing the causal relationships between variables.
    Returns:
        The Kolmogorov complexity of the inference graph.
    """
    buffer = io.BytesIO()
    nx.write_gml(inference_graph, buffer)
    return len(buffer.getvalue()) * 32 # getvalue() returns a sequence of integers in the range 0 to 255, so we multiply by 32 to get the number of bits



def causal_variable_entropy(values: List[Message]) -> float:
    distribution = {}

    for i, value in enumerate(values):
        curr_value = value['current_value']
        distribution[curr_value] = distribution.get(curr_value, 0) + 1

    entropy = 0.0
    total = sum(distribution.values())
    for count in distribution.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy

def causal_variable_conditional_entropy(values: List[Message], parent_values: Dict[str, List[Message]] = None) -> float: 
    joint_distribution = {}
    parent_distribution = {}

    for i, value in enumerate(values):
        curr_value = value['current_value']
        parent_values_i = [parent_values[parent][i]['current_value'] for parent in parent_values] # TODO: optimize
        joint_key = (curr_value, *tuple(parent_values_i))
        parent_key = tuple(parent_values_i)

        joint_distribution[joint_key] = joint_distribution.get(joint_key, 0) + 1
        parent_distribution[parent_key] = parent_distribution.get(parent_key, 0) + 1

    entropy = 0.0
    joint_total = sum(joint_distribution.values())
    parent_total = sum(parent_distribution.values())
    for key, count in joint_distribution.items():
        joint_probability = count / joint_total
        parent_probability = parent_distribution[key[1:]] / parent_total
        entropy -= joint_probability * math.log2(joint_probability / parent_probability)

    return entropy

@tool
def causal_graph_entropy(inference_graph_instantiations: List[nx.DiGraph], return_individual_entropies: bool = False) -> float:
    """
    This is a tool that computes the entropy of a causal network. The entropy of a causal network is the joint entropy of all its variables.
    It is obtained by computing the sum of the conditional entropies of each node in the graph given its parents (and standard entropy of the root nodes).

    Args:
        inference_graph_instantiations: A list of causal graphs, each representing a different instantiation of the causal network. 
                                        The structure of the causal graph is the same across all instantiations, but the values of the nodes may differ.
        return_individual_entropies: If True, the function will return a list of individual entropies for each node in the graph.
    Returns:
        The entropy of the causal network. If return_individual_entropies is True, a tuple is returned with the global entropy and a dictionary of individual entropies for each node in the graph.
    """
    entropy = 0.0
    entropies = {}
    reference_graph = inference_graph_instantiations[0]
    for node in reference_graph.nodes:
        parents = list(reference_graph.predecessors(node))
        instantiations = [graph.nodes[node] for graph in inference_graph_instantiations]
        
        if not parents:
            entropies[node] = causal_variable_entropy(instantiations)
            entropy += entropies[node]
        else:
            parent_instantiations = {parent: [graph.nodes[parent] for graph in inference_graph_instantiations] for parent in parents}
            entropies[node] = causal_variable_conditional_entropy(instantiations, parent_instantiations)
            entropy += entropies[node]
    
    if return_individual_entropies:
        return entropy, entropies
    else:
        return entropy


@tool
def observation_constrained_causal_graph_entropy(inference_graph_instantiations: List[nx.DiGraph], observations: Dict[str, Message], return_individual_entropies: bool = False, return_node_instances: bool = False) -> float:
    """
    This is a tool that computes the entropy of a causal network given a set of observations. The entropy of a causal network is the joint entropy of all its variables.
    It is obtained by computing the sum of the conditional entropies of each node in the graph given its parents (and standard entropy of the root nodes).
    The entropy is computed recursively as the set of possible values for each node depends on the values of its parents, which are constrained by the observations.

    Args:
        inference_graph_instantiations: A list of causal graphs, each representing a different instantiation of the causal network. 
                                        The structure of the causal graph is the same across all instantiations, but the values of the nodes may differ.
        observations: A dictionary of observed values for the variables in the causal network. This is used to condition the entropy calculation.
        return_individual_entropies: If True, the function will return a list of individual entropies for each node in the graph.
        return_node_instances: If True, the function will return a dictionary of node instances for each node in the graph.
    Returns:
        The entropy of the causal network. If return_individual_entropies is True, a tuple is returned with the global entropy and a dictionary of individual entropies for each node in the graph.
    """
    accepted_node_values = {node: [value] for node, value in observations.items()}
    reference_graph = inference_graph_instantiations[0]
    entropies = {}
    def _compute_entropy_rec(node: str) -> None:
        if node in accepted_node_values: # if the node is already observed, we can skip it
            return

        else:
            parents = list(reference_graph.predecessors(node))
            instantiations = [graph.nodes[node] for graph in inference_graph_instantiations]
            if not parents:
                accepted_node_values[node] = instantiations
                entropies[node] = causal_variable_entropy(instantiations)
            else:
                for parent in parents:
                    if parent not in accepted_node_values:
                        _compute_entropy_rec(parent)
                
                parent_instantiations = {parent: [None] * len(inference_graph_instantiations) for parent in parents}
                for i, graph in enumerate(inference_graph_instantiations):
                    for parent in parents:
                        if graph.nodes[parent] not in accepted_node_values[parent]:
                            for p in parents:
                                parent_instantiations[p][i] = None
                            break
                        else:
                            parent_instantiations[parent][i] = graph.nodes[parent]
                
                for i in range(len(instantiations)-1, -1, -1):
                    parent_idx_ref = list(parent_instantiations.keys())[0]
                    if parent_instantiations[parent_idx_ref][i] is None:
                        for parent in parents:
                            parent_instantiations[parent].pop(i)
                        instantiations.pop(i)

                accepted_node_values[node] = instantiations
                entropies[node] = causal_variable_conditional_entropy(instantiations, parent_instantiations)

    for node in reference_graph.nodes:
        _compute_entropy_rec(node)

    entropy = sum(entropies.values())
    result = (entropy,) + (entropies,) * return_individual_entropies + (accepted_node_values,) * return_node_instances
    return result if len(result) > 1 else result[0]


__all__ = ['kolmogorov_complexity', 'causal_graph_entropy', 'observation_constrained_causal_graph_entropy']