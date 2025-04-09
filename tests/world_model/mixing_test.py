
import networkx as nx
import pytest

from causal_world_modelling_agent.world_model.world_manager import find_mixing_coefficient_linear, find_mixing_coefficient_deviation_intervals


def build_random_uniform_world_set(graph: nx.DiGraph, num_worlds: int = 4):
    world_set = []
    for i in range(num_worlds):
        world = graph.copy()
        for node in graph.nodes():
            world.nodes[node]['name'] = node
            world.nodes[node]['current_value'] = str(i)
        world_set.append(world)
    return world_set


class TestMixingCoefficient:
    
    @pytest.fixture
    def chain_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        return G
    
    @pytest.fixture
    def diamond_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        return G
    
    @pytest.fixture
    def three_parents_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([('A', 'D'), ('B', 'D'), ('C', 'D'), ('Z', 'B'), ('Z', 'C')])
        return G
    
    @pytest.fixture
    def cycle_graph(self):
        G = nx.DiGraph()
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])
        return G
    
    def test_find_mixing_coefficient_linear_chain(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        coefficient = find_mixing_coefficient_linear(chain_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.0179) < 1e-4
    
    def test_find_mixing_coefficient_linear_chain_2(self, chain_graph):
        observations_1 = {'B'}
        observations_2 = {'A'}
        coefficient = find_mixing_coefficient_linear(chain_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.9820) < 1e-4

    def test_find_mixing_coefficient_linear_target_not_in_graph(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        pytest.raises(nx.exception.NetworkXError, find_mixing_coefficient_linear, chain_graph, 'E', observations_1, observations_2)

    def test_find_mixing_coefficient_linear_no_observations(self, chain_graph):
        coefficient = find_mixing_coefficient_linear(chain_graph, 'D', {}, {})
        assert coefficient == 0.5

    def test_find_mixing_coefficient_linear_observations_exceeded_cutoff(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        coefficient = find_mixing_coefficient_linear(chain_graph, 'D', observations_1, observations_2, traversal_cutoff=0)
        assert abs(coefficient - 0.5) < 1e-4

    def test_find_mixing_coefficient_linear_observations_non_exceeded_cutoff(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        coefficient = find_mixing_coefficient_linear(chain_graph, 'D', observations_1, observations_2, traversal_cutoff=1)
        assert abs(coefficient - 0.0179) < 1e-4

    def test_find_mixing_coefficient_linear_observations_invalid_epsilon(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        pytest.raises(ValueError, find_mixing_coefficient_linear, chain_graph, 'D', observations_1, observations_2, epsilon=0.0)

    def test_find_mixing_coefficient_linear_observations_invalid_epsilon_2(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        pytest.raises(ValueError, find_mixing_coefficient_linear, chain_graph, 'D', observations_1, observations_2, epsilon=1.0)

    def test_find_mixing_coefficient_linear_observations_modified_epsilon(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        coefficient = find_mixing_coefficient_linear(chain_graph, 'D', observations_1, observations_2, epsilon=0.5)
        assert abs(coefficient - 0.1192) < 1e-4

    def test_find_mixing_coefficient_linear_observations_not_in_graph(self, chain_graph):
        observations_1 = {'E'}
        observations_2 = {'F'}
        coefficient = find_mixing_coefficient_linear(chain_graph, 'D', observations_1, observations_2)
        assert coefficient == 0.5

    def test_find_mixing_coefficient_linear_diamond(self, diamond_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        coefficient = find_mixing_coefficient_linear(diamond_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.5) < 1e-4

    def test_find_mixing_coefficient_linear_diamond_2(self, diamond_graph):
        observations_1 = {'B'}
        observations_2 = {'C'}
        coefficient = find_mixing_coefficient_linear(diamond_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.5) < 1e-4

    def test_find_mixing_coefficient_linear_diamond_3(self, diamond_graph):
        observations_1 = {'A'}
        observations_2 = {'C'}
        coefficient = find_mixing_coefficient_linear(diamond_graph, 'B', observations_1, observations_2)
        assert abs(coefficient - 0.9820) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents(self, three_parents_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        coefficient = find_mixing_coefficient_linear(three_parents_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.5) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents_2(self, three_parents_graph):
        observations_1 = {'A'}
        observations_2 = {'B','C'}
        coefficient = find_mixing_coefficient_linear(three_parents_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.2086) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents_3(self, three_parents_graph):
        observations_1 = {'A'}
        observations_2 = {'Z'}
        coefficient = find_mixing_coefficient_linear(three_parents_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.2086) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents_4(self, three_parents_graph):
        observations_1 = {'A','B'}
        observations_2 = {'Z'}
        coefficient = find_mixing_coefficient_linear(three_parents_graph, 'D', observations_1, observations_2)
        assert abs(coefficient - 0.7914) < 1e-4

    def test_find_mixing_coefficient_linear_cycle(self, cycle_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        coefficient = find_mixing_coefficient_linear(cycle_graph, 'C', observations_1, observations_2)
        assert abs(coefficient - 0.0179) < 1e-4

    
    @pytest.fixture
    def chain_random_uniform_world_set(self, chain_graph):
        return build_random_uniform_world_set(chain_graph, num_worlds=4)
    
    @pytest.fixture
    def diamond_random_uniform_world_set(self, diamond_graph):
        return build_random_uniform_world_set(diamond_graph, num_worlds=4)
    
    @pytest.fixture
    def diamond_half_random_world_set(self, diamond_graph):
        graphs = build_random_uniform_world_set(diamond_graph, num_worlds=4)
        for i, graph in enumerate(graphs):
            graph.nodes['A']['current_value'] = '0'
            graph.nodes['C']['current_value'] = str(i // 2)
        return graphs
    
    @pytest.fixture
    def three_parents_random_uniform_world_set(self, three_parents_graph):
        return build_random_uniform_world_set(three_parents_graph, num_worlds=4)
    
    @pytest.fixture
    def cycle_random_uniform_world_set(self, cycle_graph):
        return build_random_uniform_world_set(cycle_graph, num_worlds=4)
    
    def test_find_mixing_coefficient_deviation_intervals_chain(self, chain_graph, chain_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(chain_graph, 'D', observations_1, observations_2, world_set=chain_random_uniform_world_set)
        assert abs(coefficient - 0.0498) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_chain_2(self, chain_graph, chain_random_uniform_world_set):
        observations_1 = {'B': {'name': 'B', 'current_value': '0'}}
        observations_2 = {'A': {'name': 'A', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(chain_graph, 'D', observations_1, observations_2, world_set=chain_random_uniform_world_set)
        assert abs(coefficient - 0.9502) < 1e-4
        
    def test_find_mixing_coefficient_deviation_intervals_diamond(self, diamond_graph, diamond_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(diamond_graph, 'D', observations_1, observations_2, world_set=diamond_random_uniform_world_set)
        assert abs(coefficient - 0.6164) < 1e-4
        
    def test_find_mixing_coefficient_deviation_intervals_diamond_2(self, diamond_graph, diamond_random_uniform_world_set):
        observations_1 = {'C': {'name': 'C', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(diamond_graph, 'D', observations_1, observations_2, world_set=diamond_random_uniform_world_set)
        assert abs(coefficient - 0.5) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond_half_random(self, diamond_graph, diamond_half_random_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(diamond_graph, 'D', observations_1, observations_2, world_set=diamond_half_random_world_set)
        assert abs(coefficient - 0.7840) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond_half_random_2(self, diamond_graph, diamond_half_random_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(diamond_graph, 'D', observations_1, observations_2, world_set=diamond_half_random_world_set)
        assert abs(coefficient - 0.7840) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond_half_random_3(self, diamond_graph, diamond_half_random_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'C': {'name': 'C', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(diamond_graph, 'B', observations_1, observations_2, world_set=diamond_half_random_world_set)
        assert abs(coefficient - 0.9501) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_three_parents(self, three_parents_graph, three_parents_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        coefficient = find_mixing_coefficient_deviation_intervals(three_parents_graph, 'D', observations_1, observations_2, world_set=three_parents_random_uniform_world_set)
        assert abs(coefficient - 0.5) < 1e-4

    

        