
import networkx as nx
import pytest

from causal_world_modelling_agent.tools.evaluators import causal_graph_entropy, observation_constrained_causal_graph_entropy


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
    def one_parent_half_random_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value=str(i))
            graph.add_node('B', current_value=str(i // 2))
            graph.add_edge('A', 'B')
            graphs.append(graph)
        return graphs

    @pytest.fixture
    def one_parent_half_fixed_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value=str(i // 2))
            graph.add_node('B', current_value=str(i))
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
    
    @pytest.fixture
    def four_nodes_arrow_graph_distribution(self):
        graphs = []
        for i in range(4):
            graph = nx.DiGraph()
            graph.add_node('A', current_value=str(i))
            graph.add_node('B', current_value=str(i // 2))
            graph.add_node('C', current_value=str(i // 2 + i % 2))
            graph.add_node('D', current_value=str(i // 2 + i % 2 + (4 - i)))
            graph.add_edge('A', 'C')
            graph.add_edge('B', 'C')
            graph.add_edge('C', 'D')
            graphs.append(graph)
        return graphs
    
    @pytest.fixture
    def three_nodes_arrow_graph_distribution(self):
        graphs = []
        for i in range(8):
            graph = nx.DiGraph()
            graph.add_node('A', current_value=str(i % 2))
            graph.add_node('B', current_value=str((i // 2) % 2))
            graph.add_node('C', current_value=str(i // 6))
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

    def test_one_parent_half_random_graph_entropy(self, one_parent_half_random_graph_distribution):
        entropy = causal_graph_entropy(one_parent_half_random_graph_distribution)
        assert entropy == 2.0

    def test_one_parent_half_fixed_graph_entropy(self, one_parent_half_fixed_graph_distribution):
        entropy = causal_graph_entropy(one_parent_half_fixed_graph_distribution)
        assert entropy == 2.0

    def test_two_parent_graph_entropy(self, two_parent_graph_distribution):
        entropy = causal_graph_entropy(two_parent_graph_distribution)
        assert entropy == 1.0

    def test_one_parent_individual_entropies(self, one_parent_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(one_parent_graph_distribution, return_individual_entropies=True)
        assert entropy == 2.0
        assert len(individual_entropies) == 2
        assert individual_entropies['A'] == 2.0
        assert individual_entropies['B'] == 0.0

    def test_one_parent_random_individual_entropies(self, one_parent_random_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(one_parent_random_graph_distribution, return_individual_entropies=True)
        assert entropy == 2.0
        assert len(individual_entropies) == 2
        assert individual_entropies['A'] == 0.0
        assert individual_entropies['B'] == 2.0

    def test_one_parent_fixed_individual_entropies(self, one_parent_fixed_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(one_parent_fixed_graph_distribution, return_individual_entropies=True)
        assert entropy == 2.0
        assert len(individual_entropies) == 2
        assert individual_entropies['A'] == 2.0
        assert individual_entropies['B'] == 0.0

    def test_one_parent_half_random_individual_entropies(self, one_parent_half_random_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(one_parent_half_random_graph_distribution, return_individual_entropies=True)
        assert entropy == 2.0
        assert len(individual_entropies) == 2
        assert individual_entropies['A'] == 2.0
        assert individual_entropies['B'] == 0.0

    def test_one_parent_half_fixed_individual_entropies(self, one_parent_half_fixed_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(one_parent_half_fixed_graph_distribution, return_individual_entropies=True)
        assert entropy == 2.0
        assert len(individual_entropies) == 2
        assert individual_entropies['A'] == 1.0
        assert individual_entropies['B'] == 1.0

    def test_two_parents_individual_entropies(self, two_parent_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(two_parent_graph_distribution, return_individual_entropies=True)
        assert entropy == 1.0
        assert len(individual_entropies) == 3
        assert individual_entropies['A'] == 0.0
        assert individual_entropies['B'] == 1.0
        assert individual_entropies['C'] == 0.0

    def test_four_nodes_arrow_individual_entropies(self, four_nodes_arrow_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(four_nodes_arrow_graph_distribution, return_individual_entropies=True)
        assert entropy == 3.5
        assert len(individual_entropies) == 4
        assert individual_entropies['A'] == 2.0
        assert individual_entropies['B'] == 1.0
        assert individual_entropies['C'] == 0.0
        assert individual_entropies['D'] == 0.5

    def test_three_nodes_arrow_individual_entropies(self, three_nodes_arrow_graph_distribution):
        entropy, individual_entropies = causal_graph_entropy(three_nodes_arrow_graph_distribution, return_individual_entropies=True)
        assert entropy == 2.5
        assert len(individual_entropies) == 3
        assert individual_entropies['A'] == 1.0
        assert individual_entropies['B'] == 1.0
        assert individual_entropies['C'] == 0.5

    def test_observation_constrained_uniform_graph_entropy(self, uniform_graph_distribution):
        observations = {'A': {'current_value': '0'}}
        entropy = observation_constrained_causal_graph_entropy(uniform_graph_distribution, observations)
        assert entropy == 0.0

    def test_observation_constrained_one_parent_graph_entropy(self, one_parent_graph_distribution):
        observations = {'A': {'current_value': '0'}}
        entropy = observation_constrained_causal_graph_entropy(one_parent_graph_distribution, observations)
        assert entropy == 0.0

    def test_observation_constrained_half_random_individual_entropies(self, one_parent_half_random_graph_distribution):
        observations = {'A': {'current_value': '0'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(one_parent_half_random_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 0.0
        assert len(individual_entropies) == 1
        assert individual_entropies['B'] == 0.0

    def test_observation_constrained_half_fixed_individual_entropies(self, one_parent_half_fixed_graph_distribution):
        observations = {'A': {'current_value': '0'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(one_parent_half_fixed_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 1.0
        assert len(individual_entropies) == 1
        assert individual_entropies['B'] == 1.0

    def test_observation_constrained_four_nodes_arrow_individual_entropies(self, four_nodes_arrow_graph_distribution):
        observations = {'B': {'current_value': '0'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(four_nodes_arrow_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 2.0 + 2/3 # /!\ Observation B restricts the conditining on C to worlds 0 and 1 (where B=0 and C=0 or C=1), but D is conditioned on worlds 0, 1 and 2 (where C=0 or C=1). The constrain on B disappears! TODO: assess if this is the expected behavior (Markov constraint?)
        assert len(individual_entropies) == 3
        assert individual_entropies['A'] == 2.0
        assert individual_entropies['C'] == 0.0
        assert individual_entropies['D'] == 2 / 3

    def test_observation_constrained_four_nodes_arrow_individual_entropies_2(self, four_nodes_arrow_graph_distribution):
        observations = {'A': {'current_value': '0'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(four_nodes_arrow_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 1.0
        assert len(individual_entropies) == 3
        assert individual_entropies['B'] == 1.0
        assert individual_entropies['C'] == 0.0
        assert individual_entropies['D'] == 0.0

    def test_impossible_observation(self, one_parent_half_fixed_graph_distribution):
        observations = {'A': {'current_value': '-1'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(one_parent_half_fixed_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 0.0
        assert len(individual_entropies) == 1
        assert individual_entropies['B'] == 0.0

    def test_impossible_observation_2(self, four_nodes_arrow_graph_distribution):
        observations = {'C': {'current_value': '-1'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(four_nodes_arrow_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 3.0
        assert len(individual_entropies) == 3
        assert individual_entropies['A'] == 2.0
        assert individual_entropies['B'] == 1.0
        assert individual_entropies['D'] == 0.0

    def test_impossible_observation_3(self, four_nodes_arrow_graph_distribution):
        observations = {'A': {'current_value': '0'}, 'B': {'current_value': '1'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(four_nodes_arrow_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 0.0
        assert len(individual_entropies) == 2
        assert individual_entropies['C'] == 0.0
        assert individual_entropies['D'] == 0.0

    def test_impossible_observation_4(self, three_nodes_arrow_graph_distribution):
        three_nodes_arrow_graph_distribution = three_nodes_arrow_graph_distribution[4:7]
        observations = {'A': {'current_value': '1'}, 'B': {'current_value': '1'}}
        entropy, individual_entropies = observation_constrained_causal_graph_entropy(three_nodes_arrow_graph_distribution, observations, return_individual_entropies=True)
        assert entropy == 0.0
        assert len(individual_entropies) == 1
        assert individual_entropies['C'] == 0.0
        