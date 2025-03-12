
import networkx as nx
from typing import Dict

from ..syntax.definitions import MessageDefinition, VariableDefinition, CausalRelationshipDefinition


def isMessageDefinition(data: Dict, definition: MessageDefinition) -> bool:
    try:
        definition.from_dict(data)
        return True
    except TypeError:
        return False
    
def isVariableDefinition(data: Dict) -> bool:
    return isMessageDefinition(data, VariableDefinition)

def isCausalRelationshipDefinition(data: Dict) -> bool:
    return isMessageDefinition(data, CausalRelationshipDefinition)
    

def isGraphMessageDefinition(graph: nx.DiGraph) -> bool:
    for node in graph.nodes:
        if not isMessageDefinition(graph.nodes[node], VariableDefinition):
            return False
        
    for edge in graph.edges:
        if not isMessageDefinition(graph.edges[edge], CausalRelationshipDefinition):
            return False
    