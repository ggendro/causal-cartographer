
import networkx as nx
import pytest

from causal_world_modelling_agent.utils.inference_utils import dagify


class TestDAGify:
    
    @pytest.fixture
    def no_cycle_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('A', 'C')])
        return graph
    
    @pytest.fixture
    def one_cycle_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        return graph
    
    @pytest.fixture
    def two_cycles_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'D'), ('D', 'E'), ('E', 'C')])
        return graph
    
    @pytest.fixture
    def two_mixed_cycles_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'B')])
        return graph
    
    @pytest.fixture
    def three_cycles_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'D'), ('D', 'E'), ('E', 'C'), ('E', 'F'), ('F', 'G'), ('G', 'E')])
        return graph
    
    @pytest.fixture
    def empty_graph(self):
        graph = nx.DiGraph()
        return graph
    
    @pytest.fixture
    def single_node_graph(self):
        graph = nx.DiGraph()
        graph.add_node('A')
        return graph
    
    @pytest.fixture
    def disconnected_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('C', 'D')])
        return graph
    
    @pytest.fixture
    def two_cycles_disconnected_graph(self):
        graph = nx.DiGraph()
        graph.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('D', 'E'), ('E', 'D')])
        return graph
    

    def test_dagify_no_cycle(self, no_cycle_graph):
        dag_generator = dagify(no_cycle_graph)
        dags = list(dag_generator)
        assert len(dags) == 1
        assert nx.is_directed_acyclic_graph(dags[0])
        assert dags[0].nodes == no_cycle_graph.nodes
        assert dags[0].edges == no_cycle_graph.edges

    def test_dagify_one_cycle(self, one_cycle_graph):
        dag_generator = dagify(one_cycle_graph)
        dags = list(dag_generator)
        assert len(dags) == 3
        for dag in dags:
            assert nx.is_directed_acyclic_graph(dag)
            assert dag.nodes == one_cycle_graph.nodes
            assert dag.number_of_edges() == one_cycle_graph.number_of_edges() - 1

    def test_dagify_two_cycles(self, two_cycles_graph):
        dag_generator = dagify(two_cycles_graph)
        dags = list(dag_generator)
        graph_edges = [tuple(dag.edges) for dag in dags]
        assert len(graph_edges) == len(set(graph_edges))
        assert len(dags) == 9
        for dag in dags:
            assert nx.is_directed_acyclic_graph(dag)
            assert dag.nodes == two_cycles_graph.nodes
            assert dag.number_of_edges() == two_cycles_graph.number_of_edges() - 2

    def test_dagify_two_mixed_cycles(self, two_mixed_cycles_graph):
        dag_generator = dagify(two_mixed_cycles_graph)
        dags = list(dag_generator)
        graph_edges = [tuple(dag.edges) for dag in dags]
        assert len(graph_edges) == len(set(graph_edges))
        assert len(dags) == 5

        for dag in dags:
            assert nx.is_directed_acyclic_graph(dag)
            assert dag.nodes == two_mixed_cycles_graph.nodes

        edge_count = {}
        for edges in graph_edges:
            nb_edges = len(edges)
            edge_count[nb_edges] = edge_count.get(nb_edges, 0) + 1
        assert edge_count == {2: 4, 3: 1}

    def test_dagify_three_cycles(self, three_cycles_graph):
        dag_generator = dagify(three_cycles_graph)
        dags = list(dag_generator)
        graph_edges = [tuple(dag.edges) for dag in dags]
        assert len(graph_edges) == len(set(graph_edges))
        assert len(dags) == 27
        for dag in dags:
            assert nx.is_directed_acyclic_graph(dag)
            assert dag.nodes == three_cycles_graph.nodes
            assert dag.number_of_edges() == three_cycles_graph.number_of_edges() - 3

    def test_dagify_empty_graph(self, empty_graph):
        dag_generator = dagify(empty_graph)
        dags = list(dag_generator)
        assert len(dags) == 1
        assert nx.is_directed_acyclic_graph(dags[0])
        assert dags[0].nodes == empty_graph.nodes
        assert dags[0].edges == empty_graph.edges

    def test_dagify_single_node_graph(self, single_node_graph):
        dag_generator = dagify(single_node_graph)
        dags = list(dag_generator)
        assert len(dags) == 1
        assert nx.is_directed_acyclic_graph(dags[0])
        assert dags[0].nodes == single_node_graph.nodes
        assert dags[0].edges == single_node_graph.edges

    def test_dagify_disconnected_graph(self, disconnected_graph):
        dag_generator = dagify(disconnected_graph)
        dags = list(dag_generator)
        assert len(dags) == 1
        assert nx.is_directed_acyclic_graph(dags[0])
        assert dags[0].nodes == disconnected_graph.nodes
        assert dags[0].edges == disconnected_graph.edges

    def test_dagify_two_cycles_disconnected_graph(self, two_cycles_disconnected_graph):
        dag_generator = dagify(two_cycles_disconnected_graph)
        dags = list(dag_generator)
        graph_edges = [tuple(dag.edges) for dag in dags]
        assert len(graph_edges) == len(set(graph_edges))
        assert len(dags) == 6
        for dag in dags:
            assert nx.is_directed_acyclic_graph(dag)
            assert dag.nodes == two_cycles_disconnected_graph.nodes
            assert dag.number_of_edges() == two_cycles_disconnected_graph.number_of_edges() - 2
        
    
