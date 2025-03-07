
from typing import List, Dict
import networkx as nx
import io
import math

from smolagents import tool

from ..core.definitions import Message



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

        if curr_value in distribution:
            distribution[curr_value] += 1
        else:
            distribution[curr_value] = 1

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

        if joint_key in joint_distribution:
            joint_distribution[joint_key] += 1
        else:
            joint_distribution[joint_key] = 1

        if parent_key in parent_distribution:
            parent_distribution[parent_key] += 1
        else:
            parent_distribution[parent_key] = 1

    entropy = 0.0
    joint_total = sum(joint_distribution.values())
    parent_total = sum(parent_distribution.values())
    for key, count in joint_distribution.items():
        joint_probability = count / joint_total
        parent_probability = parent_distribution[key[1:]] / parent_total
        entropy -= joint_probability * math.log2(joint_probability / parent_probability)

    return entropy

@tool
def causal_graph_entropy(inference_graph_instantiations: List[nx.DiGraph]) -> float:
    """
    This is a tool that computes the entropy of a causal network. The entropy of a causal network is the joint entropy of all its variables.
    It is obtained by computing the sum of the conditional entropies of each node in the graph given its parents (and standard entropy of the root nodes).

    Args:
        inference_graph_instantiations: A list of causal graphs, each representing a different instantiation of the causal network. 
                                        The structure of the causal graph is the same across all instantiations, but the values of the nodes may differ.
    Returns:
        The entropy of the causal network.
    """
    entropy = 0
    reference_graph = inference_graph_instantiations[0]
    for node in reference_graph.nodes:
        parents = list(reference_graph.predecessors(node))
        
        if not parents:
            instantiations = [graph.nodes[node] for graph in inference_graph_instantiations]
            entropy += causal_variable_entropy(instantiations)
        else:
            instantiations = [graph.nodes[node] for graph in inference_graph_instantiations]
            parent_instantiations = {parent: [graph.nodes[parent] for graph in inference_graph_instantiations] for parent in parents}
            entropy += causal_variable_conditional_entropy(instantiations, parent_instantiations)
    
    return entropy

