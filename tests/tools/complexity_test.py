
import networkx as nx
import pytest

from causal_world_modelling_agent.tools.evaluators import kolmogorov_complexity


class TestKolmogorovComplexity:

    @pytest.fixture
    def graph1(self):
        graph = nx.DiGraph()
        graph.add_node('A', current_value='0')
        return graph
    
    @pytest.fixture
    def graph2(self):
        graph = nx.DiGraph()
        graph.add_node('A', current_value='1')
        graph.add_node('B', current_value='0')
        graph.add_edge('A', 'B')
        return graph
    
    def test_uniform_graph_entropy(self, graph1):
        assert kolmogorov_complexity(graph1) == 2592

    def test_constant_graph_entropy(self, graph2):
        assert kolmogorov_complexity(graph2) == 5696