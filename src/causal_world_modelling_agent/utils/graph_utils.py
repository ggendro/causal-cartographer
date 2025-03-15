from typing import Any, List, Optional
import networkx as nx

def isGraph(answer: Any, memory: Optional[List[Any]] = None) -> bool:
    if not isinstance(answer, nx.Graph):
        raise TypeError(f"The answer is not a networkx graph ({type(nx.Graph)}). Data type: {type(answer)}")

    return True

def isDigraph(answer: Any, memory: Optional[List[Any]] = None) -> bool:
    if not isinstance(answer, nx.DiGraph):
        raise TypeError(f"The answer is not a networkx digraph ({type(nx.DiGraph)}). Data type: {type(answer)}")

    return True

def isDAG(answer: Any, memory: Optional[List[Any]] = None) -> bool:
    if not isDigraph(answer) and nx.is_directed_acyclic_graph(answer):
        raise ValueError("The answer is not a directed acyclic graph (DAG)")
    
    return True
