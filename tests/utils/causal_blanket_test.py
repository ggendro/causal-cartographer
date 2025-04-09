
import networkx as nx
import pytest

from causal_world_modelling_agent.utils.inference_utils import build_target_node_causal_blanket


class TestCausalBlanket:
    @pytest.fixture
    def simple_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('A', 'C')])
        return graph
    
    @pytest.fixture
    def diamond_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        return graph
    
    @pytest.fixture
    def disconnected_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('C', 'D')])
        return graph
    
    @pytest.fixture
    def cyclic_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
        return graph
    
    @pytest.fixture
    def long_chain_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')])
        return graph
    
    
    def test_simple_graph(self, simple_graph):
        blanket = build_target_node_causal_blanket(simple_graph, 'C', ['A'])
        assert blanket == {'A', 'C'}

    def test_simple_graph_no_parents(self, simple_graph):
        blanket = build_target_node_causal_blanket(simple_graph, 'C', [])
        assert blanket == {'A', 'C'}

    def test_simple_graph_confounder(self, simple_graph):
        blanket = build_target_node_causal_blanket(simple_graph, 'C', ['B'])
        assert blanket == {'A', 'C'}

    def test_simple_graph_known_target(self, simple_graph):
        blanket = build_target_node_causal_blanket(simple_graph, 'C', ['C'])
        assert blanket == {'C'}

    def test_diamond_graph(self, diamond_graph):
        blanket = build_target_node_causal_blanket(diamond_graph, 'D', ['A'])
        assert blanket == {'A', 'B', 'C', 'D'}

    def test_diamond_graph_intermediate_observations(self, diamond_graph):
        blanket = build_target_node_causal_blanket(diamond_graph, 'D', ['B', 'C'])
        assert blanket == {'B', 'C', 'D'}

    def test_diamond_graph_no_parents(self, diamond_graph):
        blanket = build_target_node_causal_blanket(diamond_graph, 'D', [])
        assert blanket == {'A', 'B', 'C', 'D'}

    def test_disconnected_graph(self, disconnected_graph):
        blanket = build_target_node_causal_blanket(disconnected_graph, 'D', ['A'])
        assert blanket == {'C', 'D'}

    def test_disconnected_graph_no_parents(self, disconnected_graph):
        blanket = build_target_node_causal_blanket(disconnected_graph, 'D', [])
        assert blanket == {'C', 'D'}

    def test_cyclic_graph(self, cyclic_graph):
        blanket = build_target_node_causal_blanket(cyclic_graph, 'D', [])
        assert blanket == {'A', 'B', 'C', 'D'}

    def test_cyclic_graph_with_parents(self, cyclic_graph):
        blanket = build_target_node_causal_blanket(cyclic_graph, 'D', ['B'])
        assert blanket == {'B', 'C', 'D'}

    def test_long_chain_graph(self, long_chain_graph):
        blanket = build_target_node_causal_blanket(long_chain_graph, 'E', ['C'])
        assert blanket == {'C', 'D', 'E'}

    def test_long_chain_graph_no_parents(self, long_chain_graph):
        blanket = build_target_node_causal_blanket(long_chain_graph, 'E', [])
        assert blanket == {'A', 'B', 'C', 'D', 'E'}

    def test_long_chain_graph_with_cutoff(self, long_chain_graph):
        pytest.raises(nx.exception.ExceededMaxIterations, build_target_node_causal_blanket, long_chain_graph, 'E', ['B'], traversal_cutoff=3)

    def test_long_chain_graph_with_cutoff_2(self, long_chain_graph):
        blanket = build_target_node_causal_blanket(long_chain_graph, 'E', ['C'], traversal_cutoff=3)
        assert blanket == {'C', 'D', 'E'}

