
from typing import Any
import networkx as nx


def is_graph(data: Any, *args, **kwargs) -> bool:
    return isinstance(data, nx.Graph)

def is_digraph(data: Any, *args, **kwargs) -> bool:
    print(data)
    return isinstance(data, nx.DiGraph)

def is_dag(data: Any, *args, **kwargs) -> bool:
    return is_digraph(data) and nx.is_directed_acyclic_graph(data)