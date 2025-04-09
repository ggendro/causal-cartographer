
import networkx as nx
import pytest

from causal_world_modelling_agent.utils.inference_utils import convert_path_to_chain, find_causal_paths


class TestConvertPathToChain:

    @pytest.fixture
    def simple_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        return graph
    
    @pytest.fixture
    def simple_chain(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        return graph

    @pytest.fixture
    def complex_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('E', 'F')])
        return graph
    
    @pytest.fixture
    def complex_chain(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'D'), ('D', 'E')])
        return graph
    
    @pytest.fixture
    def graph_with_cycles(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A'), ('A', 'E')])
        return graph
    
    @pytest.fixture
    def chain_with_cycles(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')])
        return graph
    
    @pytest.fixture
    def graph_with_duplicate_paths(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('D' 'F'), ('E', 'G'), ('F', 'G'), ('A', 'D'), ('D', 'G'), ('A', 'G')])
        return graph
    
    @pytest.fixture
    def chain_with_duplicate_paths(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'D'), ('D', 'E'), ('E', 'G'), ('F', 'G'), ('D', 'F'), ('C', 'D'), ('A', 'C')])
        return graph
    
    @pytest.fixture
    def disconnected_chain(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('C', 'D')])
        return graph
    
    @pytest.fixture
    def graph_with_isolated_nodes(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('C', 'D')])
        return graph
    
    
    def test_convert_path_to_chain(self, simple_graph, simple_chain):
        path = ['A', 'B', 'C']
        chain = convert_path_to_chain(path, simple_graph)

        assert chain.edges() == simple_chain.edges()
        assert chain.nodes() == simple_chain.nodes()

    def test_convert_path_to_chain_complex(self, complex_graph, complex_chain):
        path = ['A', 'B', 'D', 'E']
        chain = convert_path_to_chain(path, complex_graph)

        assert chain.edges() == complex_chain.edges()
        assert chain.nodes() == complex_chain.nodes()

    def test_convert_path_to_chain_with_cycles(self, graph_with_cycles, chain_with_cycles):
        path = ['A', 'B', 'C', 'D', 'A']
        chain = convert_path_to_chain(path, graph_with_cycles)

        assert chain.edges() == chain_with_cycles.edges()
        assert chain.nodes() == chain_with_cycles.nodes()

    def test_convert_path_to_chain_with_duplicate_paths(self, graph_with_duplicate_paths, chain_with_duplicate_paths):
        path = ['A', 'B', 'D', 'E', 'G', 'F', 'D', 'C', 'A']
        chain = convert_path_to_chain(path, graph_with_duplicate_paths)

        assert chain.edges() == chain_with_duplicate_paths.edges()
        assert chain.nodes() == chain_with_duplicate_paths.nodes()

    def test_convert_path_to_chain_disconnected(self, disconnected_chain, complex_graph):
        path = ['A', 'B', 'C', 'D']
        pytest.raises(ValueError, convert_path_to_chain, path, disconnected_chain)

    def test_convert_path_to_chain_with_isolated_nodes(self, graph_with_isolated_nodes):
        path = ['A', 'B', 'C', 'D']
        pytest.raises(ValueError, convert_path_to_chain, path, graph_with_isolated_nodes)



class TestFindCausalPaths:

    @pytest.fixture
    def simple_causal_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C')])
        return graph

    @pytest.fixture
    def diamond_causal_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        return graph

    @pytest.fixture
    def long_chain_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        return graph

    def test_simple_path_no_conditioning(self, simple_causal_graph):
        paths = find_causal_paths(simple_causal_graph, 'A', 'C', [])
        assert paths == [['A', 'B', 'C']]

    def test_simple_path_with_conditioning_dseparation(self, simple_causal_graph):
        paths = find_causal_paths(simple_causal_graph, 'A', 'C', ['B'])
        assert paths == []

    def test_diamond_paths_no_conditioning(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'A', 'D', [])
        expected = [['A', 'B', 'D'], ['A', 'C', 'D']]
        assert sorted(paths) == sorted(expected)

    def test_diamond_path_with_conditioning(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'A', 'D', ['B'])
        assert paths == [['A', 'C', 'D']]

    def test_traversal_cutoff_1(self, long_chain_graph):
        paths = find_causal_paths(long_chain_graph, 'A', 'D', [], traversal_cutoff=2)
        assert paths == []

    def test_traversal_cutoff_2(self, long_chain_graph):
        paths = find_causal_paths(long_chain_graph, 'A', 'D', [], traversal_cutoff=4)
        assert paths == [['A', 'B', 'C', 'D']]

    def test_reverse_path(self, simple_causal_graph):
        paths = find_causal_paths(simple_causal_graph, 'C', 'A', [])
        assert paths == [['C', 'B', 'A']]

    def test_conditioning_set_not_in_chain(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'A', 'D', ['X'])
        expected = [['A', 'B', 'D'], ['A', 'C', 'D']]
        assert sorted(paths) == sorted(expected)

    def test_blocked_path(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'A', 'D', ['B', 'C'])
        assert paths == []

    def test_backdoor_path(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'B', 'C', [])
        assert paths == [['B', 'A', 'C']]

    def test_blocked_backdoor_path(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'B', 'C', ['A'])
        assert paths == []

    def test_collider_path(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'B', 'C', ['A', 'D'])
        assert paths == [['B', 'D', 'C']]

    def test_direct_path(self, simple_causal_graph):
        paths = find_causal_paths(simple_causal_graph, 'A', 'B', [])
        assert paths == [['A', 'B']]

    def test_direct_and_backdoor_path(self, diamond_causal_graph):
        paths = find_causal_paths(diamond_causal_graph, 'B', 'D', [])
        expected = [['B', 'D'], ['B', 'A', 'C', 'D']]
        assert sorted(paths) == sorted(expected)

