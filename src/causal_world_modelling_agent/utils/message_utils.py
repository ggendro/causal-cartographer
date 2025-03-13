
import networkx as nx
from typing import Dict, List, Any

from ..syntax.definitions import MessageDefinition, VariableDefinition, CausalRelationshipDefinition


def isMessageDefinition(data: Dict, definition: MessageDefinition) -> bool:
    try:
        definition.from_dict(data)
        return True
    except TypeError as e: # TODO: modify excpetion logging to be more informative
        raise e
    
def isVariableDefinition(answer: Dict, memory: List[Any]) -> bool:
    return isMessageDefinition(answer, VariableDefinition)

def isCausalRelationshipDefinition(answer: dict, memory: List[Any]) -> bool:
    return isMessageDefinition(answer, CausalRelationshipDefinition)
    

def isGraphMessageDefinition(graph: nx.DiGraph, memory: List[Any]) -> bool:
    for node in graph.nodes:
        isMessageDefinition(graph.nodes[node], VariableDefinition)
    
    for edge in graph.edges:
        isMessageDefinition(graph.edges[edge], CausalRelationshipDefinition)
    return True
    