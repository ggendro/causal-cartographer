
from typing import Any
import networkx as nx


def isGraph(data: Any, *args, **kwargs) -> bool:
    return isinstance(data, nx.Graph)

def isDigraph(data: Any, *args, **kwargs) -> bool:
    print(data)
    return isinstance(data, nx.DiGraph)

def isDAG(data: Any, *args, **kwargs) -> bool:
    return isDigraph(data) and nx.is_directed_acyclic_graph(data)