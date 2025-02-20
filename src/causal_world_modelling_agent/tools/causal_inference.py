
from typing import List, Tuple
import networkx as nx

from smolagents import tool, MultiStepAgent

from ..core.definitions import Message




@tool
def serialize_graph(graph: nx.DiGraph) -> List[Tuple[Message, Message]]:
    """
    Serialize a graph into a list of tuples.
    
    Args:
        graph: The graph to serialize.
    """
    return list(graph.edges())

@tool
def deserialize_graph(serialized_graph: List[Tuple[Message, Message]]) -> nx.DiGraph:
    """
    Deserialize a list of tuples into a graph.
    
    Args:
        serialized_graph: The list of tuples to deserialize.
    """
    graph = nx.DiGraph()
    for u, v in serialized_graph:
        graph.add_edge(u, v)
    return graph

@tool
def causal_graph_inference(graph: nx.DiGraph, agent: MultiStepAgent) -> nx.DiGraph:
    """
    Infer the causal graph from the given graph.
    
    Args:
        graph: The graph to infer the causal graph from.
        agent: The agent to use for inference.
    """
    return agent.infer(graph)
