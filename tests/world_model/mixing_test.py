import networkx as nx
import pytest

from causal_world_modelling_agent.world_model.world_manager import (
    find_mixing_coefficient_linear, 
    find_mixing_coefficient_deviation_intervals,
    find_mixing_coefficients_overlapping_beams
)


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
    
    # Linear coefficient tests

    def test_find_mixing_coefficient_linear_chain(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.0179
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_chain_2(self, chain_graph):
        observations_1 = {'B'}
        observations_2 = {'A'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.9820
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_target_not_in_graph(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        with pytest.raises(nx.exception.NetworkXError):
            find_mixing_coefficient_linear(
                chain_graph, 'E', observations_1, observations_2,
                factual_target_value='0', counterfactual_target_value='1'
            )

    def test_find_mixing_coefficient_linear_no_observations(self, chain_graph):
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            chain_graph, 'D', {}, {},
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.5
        assert result[factual] == expected
        assert result[counterfactual] == 1 - expected

    def test_find_mixing_coefficient_linear_observations_exceeded_cutoff(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            traversal_cutoff=0
        )
        expected = 0.5
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_observations_non_exceeded_cutoff(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            traversal_cutoff=1
        )
        expected = 0.0179
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_observations_invalid_epsilon(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        with pytest.raises(ValueError):
            find_mixing_coefficient_linear(
                chain_graph, 'D', observations_1, observations_2,
                factual_target_value='0', counterfactual_target_value='1',
                epsilon=0.0
            )

    def test_find_mixing_coefficient_linear_observations_invalid_epsilon_2(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        with pytest.raises(ValueError):
            find_mixing_coefficient_linear(
                chain_graph, 'D', observations_1, observations_2,
                factual_target_value='0', counterfactual_target_value='1',
                epsilon=1.0
            )

    def test_find_mixing_coefficient_linear_observations_modified_epsilon(self, chain_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            epsilon=0.5
        )
        expected = 0.1192
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_observations_not_in_graph(self, chain_graph):
        observations_1 = {'E'}
        observations_2 = {'F'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.5
        assert result[factual] == expected
        assert result[counterfactual] == 1 - expected

    def test_find_mixing_coefficient_linear_diamond(self, diamond_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            diamond_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.5
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_diamond_2(self, diamond_graph):
        observations_1 = {'B'}
        observations_2 = {'C'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            diamond_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.5
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_diamond_3(self, diamond_graph):
        observations_1 = {'A'}
        observations_2 = {'C'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            diamond_graph, 'B', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.9820
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents(self, three_parents_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            three_parents_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.5
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents_2(self, three_parents_graph):
        observations_1 = {'A'}
        observations_2 = {'B', 'C'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            three_parents_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.2086
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents_3(self, three_parents_graph):
        observations_1 = {'A'}
        observations_2 = {'Z'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            three_parents_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.2086
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_three_parents_4(self, three_parents_graph):
        observations_1 = {'A', 'B'}
        observations_2 = {'Z'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            three_parents_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.7914
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_linear_cycle(self, cycle_graph):
        observations_1 = {'A'}
        observations_2 = {'B'}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_linear(
            cycle_graph, 'C', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual
        )
        expected = 0.0179
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    # Fixtures for world sets
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
    def three_parents_random_uniform_world_set_half_random(self, three_parents_graph):
        graphs = build_random_uniform_world_set(three_parents_graph, num_worlds=4)
        for i, graph in enumerate(graphs):
            graph.nodes['A']['current_value'] = '0'
            graph.nodes['B']['current_value'] = str(i // 2)
        return graphs
    
    # Deviation intervals tests

    def test_find_mixing_coefficient_deviation_intervals_chain(self, chain_graph, chain_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=chain_random_uniform_world_set
        )
        expected = 0.0498
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_chain_2(self, chain_graph, chain_random_uniform_world_set):
        observations_1 = {'B': {'name': 'B', 'current_value': '0'}}
        observations_2 = {'A': {'name': 'A', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            chain_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=chain_random_uniform_world_set
        )
        expected = 0.9502
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond(self, diamond_graph, diamond_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            diamond_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=diamond_random_uniform_world_set
        )
        expected = 0.6164
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond_2(self, diamond_graph, diamond_random_uniform_world_set):
        observations_1 = {'C': {'name': 'C', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            diamond_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=diamond_random_uniform_world_set
        )
        expected = 0.5
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond_half_random(self, diamond_graph, diamond_half_random_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            diamond_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=diamond_half_random_world_set
        )
        expected = 0.7840
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond_half_random_2(self, diamond_graph, diamond_half_random_world_set):
        # Repeated test case with the same parameters
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            diamond_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=diamond_half_random_world_set
        )
        expected = 0.7840
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_diamond_half_random_3(self, diamond_graph, diamond_half_random_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'C': {'name': 'C', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            diamond_graph, 'B', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=diamond_half_random_world_set
        )
        expected = 0.9501
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    def test_find_mixing_coefficient_deviation_intervals_three_parents(self, three_parents_graph, three_parents_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        factual = '0'
        counterfactual = '1'
        result = find_mixing_coefficient_deviation_intervals(
            three_parents_graph, 'D', observations_1, observations_2,
            factual_target_value=factual, counterfactual_target_value=counterfactual,
            world_set=three_parents_random_uniform_world_set
        )
        expected = 0.5
        assert abs(result[factual] - expected) < 1e-4
        assert abs(result[counterfactual] - (1 - expected)) < 1e-4

    # Overlapping beams tests

    def test_find_mixing_coefficients_overlapping_beams_chain(self, chain_graph, chain_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        result = find_mixing_coefficients_overlapping_beams(
            chain_graph, 'D', observations_1, observations_2,
            world_set=chain_random_uniform_world_set
        )
        expected = {'0': 0.4, '1': 0.2, '2': 0.2, '3': 0.2}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_chain_higher_temperature(self, chain_graph, chain_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        result = find_mixing_coefficients_overlapping_beams(
            chain_graph, 'D', observations_1, observations_2,
            world_set=chain_random_uniform_world_set, temperature=10.0
        )
        expected = {'0': 0.2632, '1': 0.2456, '2': 0.2456, '3': 0.2456}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_chain_lower_temperature(self, chain_graph, chain_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        result = find_mixing_coefficients_overlapping_beams(
            chain_graph, 'D', observations_1, observations_2,
            world_set=chain_random_uniform_world_set, temperature=0.5
        )
        expected = {'0': 0.5714, '1': 0.1429, '2': 0.1429, '3': 0.1429}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_diamond(self, diamond_graph, diamond_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        result = find_mixing_coefficients_overlapping_beams(
            diamond_graph, 'D', observations_1, observations_2,
            world_set=diamond_random_uniform_world_set
        )
        expected = {'0': 0.4, '1': 0.2, '2': 0.2, '3': 0.2}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_diamond_2(self, diamond_graph, diamond_random_uniform_world_set):
        observations_1 = {'C': {'name': 'C', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        result = find_mixing_coefficients_overlapping_beams(
            diamond_graph, 'D', observations_1, observations_2,
            world_set=diamond_random_uniform_world_set
        )
        expected = {'0': 0.4, '1': 0.2, '2': 0.2, '3': 0.2}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_diamond_half_random(self, diamond_graph, diamond_half_random_world_set):
        observations_1 = {}
        observations_2 = {'C': {'name': 'C', 'current_value': '0'}}
        result = find_mixing_coefficients_overlapping_beams(
            diamond_graph, 'D', observations_1, observations_2,
            world_set=diamond_half_random_world_set
        )
        expected = {'0': 1/3, '1': 1/3, '2': 1/6, '3': 1/6}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_diamond_half_random_2(self, diamond_graph, diamond_half_random_world_set):
        observations_1 = {}
        observations_2 = {}
        result = find_mixing_coefficients_overlapping_beams(
            diamond_graph, 'D', observations_1, observations_2,
            world_set=diamond_half_random_world_set
        )
        expected = {'0': 1/4, '1': 1/4, '2': 1/4, '3': 1/4}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_three_parents(self, three_parents_graph, three_parents_random_uniform_world_set):
        observations_1 = {'A': {'name': 'A', 'current_value': '1'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '2'}}
        result = find_mixing_coefficients_overlapping_beams(
            three_parents_graph, 'D', observations_1, observations_2,
            world_set=three_parents_random_uniform_world_set
        )
        expected = {'0': 0.25, '1': 0.25, '2': 0.25, '3': 0.25}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_three_parents_half_random(self, three_parents_graph, three_parents_random_uniform_world_set_half_random):
        observations_1 = {'A': {'name': 'A', 'current_value': '0'}}
        observations_2 = {'B': {'name': 'B', 'current_value': '0'}}
        result = find_mixing_coefficients_overlapping_beams(
            three_parents_graph, 'D', observations_1, observations_2,
            world_set=three_parents_random_uniform_world_set_half_random
        )
        expected = {'0': 1/3, '1': 1/3, '2': 1/6, '3': 1/6}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())

    def test_find_mixing_coefficients_overlapping_beams_three_parents_half_random_2(self, three_parents_graph, three_parents_random_uniform_world_set_half_random):
        observations_1 = {}
        observations_2 = {'B': {'name': 'B', 'current_value': '1'}}
        result = find_mixing_coefficients_overlapping_beams(
            three_parents_graph, 'D', observations_1, observations_2,
            world_set=three_parents_random_uniform_world_set_half_random
        )
        expected = {'0': 1/6, '1': 1/6, '2': 1/3, '3': 1/3}
        assert len(result) == len(expected)
        assert all(abs(result[key] - expected[key]) < 1e-4 for key in expected.keys())
    