
import networkx as nx
import pytest

from causal_world_modelling_agent.tools.evaluators import causal_graph_entropy


class TestGraphEntropy:

    @pytest.fixture
    def uniform_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value=str(i))
            graphs.append(graph)
        return graphs
    
    @pytest.fixture
    def constant_graph_distribution(self):
        graphs = []
        for _ in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value='0')
            graphs.append(graph)
        return graphs
    
    @pytest.fixture
    def one_parent_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value=str(i))
            graph.add_node('B', current_value=str(i))
            graph.add_edge('A', 'B')
            graphs.append(graph)
        return graphs
    
    @pytest.fixture
    def one_parent_random_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value='0')
            graph.add_node('B', current_value=str(i))
            graph.add_edge('A', 'B')
            graphs.append(graph)
        return graphs
    
    @pytest.fixture
    def one_parent_fixed_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value=str(i))
            graph.add_node('B', current_value='0')
            graph.add_edge('A', 'B')
            graphs.append(graph)
        return graphs
    
    @pytest.fixture
    def two_parent_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value='0')
            graph.add_node('B', current_value=str(i // 2))
            graph.add_node('C', current_value=str(i // 2))
            graph.add_edge('A', 'C')
            graph.add_edge('B', 'C')
            graphs.append(graph)
        return graphs
    
    def test_uniform_graph_entropy(self, uniform_graph_distribution):
        entropy = causal_graph_entropy(uniform_graph_distribution)
        assert entropy == 2.0

    def test_constant_graph_entropy(self, constant_graph_distribution):
        entropy = causal_graph_entropy(constant_graph_distribution)
        assert entropy == 0.0

    def test_one_parent_graph_entropy(self, one_parent_graph_distribution):
        entropy = causal_graph_entropy(one_parent_graph_distribution)
        assert entropy == 2.0

    def test_one_parent_random_graph_entropy(self, one_parent_random_graph_distribution):
        entropy = causal_graph_entropy(one_parent_random_graph_distribution)
        assert entropy == 2.0

    def test_one_parent_fixed_graph_entropy(self, one_parent_fixed_graph_distribution):
        entropy = causal_graph_entropy(one_parent_fixed_graph_distribution)
        assert entropy == 2.0

    def test_two_parent_graph_entropy(self, two_parent_graph_distribution):
        entropy = causal_graph_entropy(two_parent_graph_distribution)
        assert entropy == 1.0