
import networkx as nx
import pytest

from causal_world_modelling_agent.world_model.world_manager import find_shared_observations, find_active_interventions, find_non_blocking_observations


class TestFindInterventions:

    def test_find_shared_observations(self):
        observations_1 = {
            'observation_1': {'current_value': 'value_1'},
            'observation_2': {'current_value': 'value_2'},
            'observation_3': {'current_value': 'value_3'}
        }
        observations_2 = {
            'observation_1': {'current_value': 'value_1'},
            'observation_2': {'current_value': 'value_2'},
            'observation_3': {'current_value': 'value_4'},
            'observation_4': {'current_value': 'value_3'}
        }
        expected_shared_observations = {
            'observation_1': {'current_value': 'value_1'},
            'observation_2': {'current_value': 'value_2'}
        }
        shared_observations = find_shared_observations(observations_1, observations_2)
        assert shared_observations == expected_shared_observations

    def test_find_shared_observations_no_current_value_attributes_raise(self):
        observations_1 = {
            'observation_1': {'current_value': 'value_1'},
            'observation_2': {'current_valuex': 'value_2'},
        }
        observations_2 = {
            'observation_1': {'current_value': 'value_1'},
            'observation_2': {'current_value': 'value_2'},
            'observation_3': {'current_value': 'value_4'}
        }
        pytest.raises(KeyError, find_shared_observations, observations_1, observations_2)

    def test_find_shared_observations_no_current_value_attributes_no_raise(self):
        observations_1 = {
            'observation_1': {'current_value': 'value_1'},
        }
        observations_2 = {
            'observation_1': {'current_value': 'value_1'},
            'observation_2': {'current_valuex': 'value_2'},
            'observation_3': {'current_value': 'value_4'}
        }
        shared_observations = find_shared_observations(observations_1, observations_2)
        assert shared_observations == {'observation_1': {'current_value': 'value_1'}}

    
    @pytest.fixture
    def chain_graph(self):
        G = nx.DiGraph()
        G.add_node('A', current_value='value_A')
        G.add_node('B', current_value='value_B')
        G.add_node('C', current_value='value_C')
        G.add_node('D', current_value='value_D')
        G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'D')])
        return G
    
    @pytest.fixture
    def diamond_graph(self):
        G = nx.DiGraph()
        G.add_node('A', current_value='value_A', name='A')
        G.add_node('B', current_value='value_B', name='B')
        G.add_node('C', current_value='value_C', name='C')
        G.add_node('D', current_value='value_D', name='D')
        G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D')])
        return G

    def test_find_active_interventions_chain_graph(self, chain_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        active_interventions = find_active_interventions(chain_graph, 'D', observations, interventions)
        assert active_interventions == {'C': {'current_value': 'value_C', 'name': 'C'}}

    def test_find_active_interventions_chain_graph_2(self, chain_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'C': {'current_value': 'value_C', 'name': 'C'},
            'D': {'current_value': 'value_D', 'name': 'D'},
        }
        active_interventions = find_active_interventions(chain_graph, 'D', observations, interventions)
        assert active_interventions == {'D': {'current_value': 'value_D', 'name': 'D'}}

    def test_find_active_interventions_chain_graph_3(self, chain_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'D': {'current_value': 'value_D', 'name': 'D'},
        }
        active_interventions = find_active_interventions(chain_graph, 'C', observations, interventions)
        assert active_interventions == {}

    def test_find_active_interventions_chain_graph__4(self, chain_graph):
        observations = {
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'B': {'current_value': 'value_B', 'name': 'B'},
            'D': {'current_value': 'value_D', 'name': 'D'},
        }
        active_interventions = find_active_interventions(chain_graph, 'C', observations, interventions)
        assert active_interventions == {}

    def test_find_active_interventions_chain_graph_target_not_in_graph(self, chain_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'C': {'current_value': 'value_C', 'name': 'C'},
            'D': {'current_value': 'value_D', 'name': 'D'},
        }
        pytest.raises(ValueError, find_active_interventions, chain_graph, 'E', observations, interventions)

    def test_find_active_interventions_chain_observation_not_in_graph(self, chain_graph):
        observations = {
            'E': {'current_value': 'value_E', 'name': 'E'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        active_interventions = find_active_interventions(chain_graph, 'D', observations, interventions)
        assert active_interventions == {'A': {'current_value': 'value_A', 'name': 'A'}}

    def test_find_active_interventions_chain_intervention_not_in_graph(self, chain_graph):
        observations = {}
        interventions = {
            'E': {'current_value': 'value_E', 'name': 'E'},
        }
        pytest.raises(nx.exception.NodeNotFound, find_active_interventions, chain_graph, 'D', observations, interventions)

    def test_find_active_interventions_chain_exceded_cutoff(self, chain_graph):
        observations = {}
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        active_interventions = find_active_interventions(chain_graph, 'D', observations, interventions, traversal_cutoff=2)
        assert active_interventions == {}

    def test_find_active_interventions_chain_not_exceded_cutoff(self, chain_graph):
        observations = {}
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        active_interventions = find_active_interventions(chain_graph, 'D', observations, interventions, traversal_cutoff=3)
        assert active_interventions == {'A': {'current_value': 'value_A', 'name': 'A'}}

    def test_find_active_interventions_diamond_graph(self, diamond_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'D', observations, interventions)
        assert active_interventions == {}

    def test_find_active_interventions_diamond_graph_2(self, diamond_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'D', observations, interventions)
        assert active_interventions == {'A': {'current_value': 'value_A', 'name': 'A'}}

    def test_find_active_interventions_diamond_graph_3(self, diamond_graph):
        observations = {
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'D', observations, interventions)
        assert active_interventions == {'A': {'current_value': 'value_A', 'name': 'A'}}

    def test_find_active_interventions_diamond_graph_4(self, diamond_graph):
        observations = {}
        interventions = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'D', observations, interventions)
        assert active_interventions == {'B': {'current_value': 'value_B', 'name': 'B'}}

    def test_find_active_interventions_diamond_graph_5(self, diamond_graph):
        observations = {
            'D': {'current_value': 'value_D', 'name': 'D'},
            }
        interventions = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'C', observations, interventions)
        assert active_interventions == {'B': {'current_value': 'value_B', 'name': 'B'}}

    def test_find_active_interventions_diamond_graph_6(self, diamond_graph):
        observations = {}
        interventions = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'C', observations, interventions)
        assert active_interventions == {}

    def test_find_active_interventions_diamond_graph_7(self, diamond_graph):
        observations = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'D': {'current_value': 'value_D', 'name': 'D'},
            }
        interventions = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'C', observations, interventions)
        assert active_interventions == {'B': {'current_value': 'value_B', 'name': 'B'}}

    def test_find_active_interventions_diamond_graph_8(self, diamond_graph):
        observations = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            }
        interventions = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        active_interventions = find_active_interventions(diamond_graph, 'C', observations, interventions)
        assert active_interventions == {}


    def test_find_non_blocking_observations_chain_graph(self, chain_graph):
        observations = {
            'A': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {}
        blocking_observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        non_blocking_observations = find_non_blocking_observations(chain_graph, 'D', observations, blocking_observations, interventions)
        assert non_blocking_observations == {'A': {'current_value': 'value_B', 'name': 'B'}}
        
    def test_find_non_blocking_observations_chain_graph_2(self, chain_graph):
        observations = {
            'A': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {}
        blocking_observations = {}
        non_blocking_observations = find_non_blocking_observations(chain_graph, 'D', observations, blocking_observations, interventions)
        assert non_blocking_observations == {}
        
    def test_find_non_blocking_observations_chain_graph_3(self, chain_graph):
        observations = {
            'A': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        blocking_observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        non_blocking_observations = find_non_blocking_observations(chain_graph, 'D', observations, blocking_observations, interventions)
        assert non_blocking_observations == {}
        
    def test_find_non_blocking_observations_chain_graph_4(self, chain_graph):
        observations = {
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        interventions = {
            'D': {'current_value': 'value_D', 'name': 'D'},
        }
        blocking_observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        non_blocking_observations = find_non_blocking_observations(chain_graph, 'A', observations, blocking_observations, interventions)
        assert non_blocking_observations == {'C': {'current_value': 'value_C', 'name': 'C'}}
        
    def test_find_non_blocking_observations_chain_graph_5(self, chain_graph):
        observations = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {}
        blocking_observations = {
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        non_blocking_observations = find_non_blocking_observations(chain_graph, 'D', observations, blocking_observations, interventions)
        assert non_blocking_observations == {'A': {'current_value': 'value_A', 'name': 'A'}, 'B': {'current_value': 'value_B', 'name': 'B'}}

    def test_find_non_blocking_observations_diamond_graph(self, diamond_graph):
        observations = {
            'A': {'current_value': 'value_A', 'name': 'A'},
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {}
        blocking_observations = {
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        non_blocking_observations = find_non_blocking_observations(diamond_graph, 'D', observations, blocking_observations, interventions)
        assert non_blocking_observations == {}

    def test_find_non_blocking_observations_diamond_graph_2(self, diamond_graph):
        observations = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        interventions = {}
        blocking_observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
            'C': {'current_value': 'value_C', 'name': 'C'},
        }
        non_blocking_observations = find_non_blocking_observations(diamond_graph, 'D', observations, blocking_observations, interventions)
        assert non_blocking_observations == {'A': {'current_value': 'value_A', 'name': 'A'}}

    def test_find_non_blocking_observations_diamond_graph_3(self, diamond_graph):
        observations = {
            'A': {'current_value': 'value_A', 'name': 'A'},
        }
        interventions = {
            'C': {'current_value': 'value_C', 'name': 'C'}
        }
        blocking_observations = {
            'B': {'current_value': 'value_B', 'name': 'B'}
        }
        non_blocking_observations = find_non_blocking_observations(diamond_graph, 'D', observations, blocking_observations, interventions)
        assert non_blocking_observations == {'A': {'current_value': 'value_A', 'name': 'A'}}

    def test_find_non_blocking_observations_diamond_graph_4(self, diamond_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
        }
        interventions = {
            'C': {'current_value': 'value_C', 'name': 'C'}
        }
        blocking_observations = {
            'D': {'current_value': 'value_D', 'name': 'D'}
        }
        non_blocking_observations = find_non_blocking_observations(diamond_graph, 'C', observations, blocking_observations, interventions)
        assert non_blocking_observations == {}

    def test_find_non_blocking_observations_diamond_graph_5(self, diamond_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
            'D': {'current_value': 'value_D', 'name': 'D'}
        }
        interventions = {
            'A': {'current_value': 'value_A', 'name': 'A'}
        }
        blocking_observations = {
            'C': {'current_value': 'value_C', 'name': 'C'}
        }
        non_blocking_observations = find_non_blocking_observations(diamond_graph, 'A', observations, blocking_observations, interventions)
        assert non_blocking_observations == {} #  Observations are not mutually considered although they can create collider paths. This should not affect the downstream tasks since out-edges of the target are cut. TODO: assess if this is the expected behavior. 

    def test_find_non_blocking_observations_non_disjoint_sets(self, diamond_graph):
        observations = {
            'B': {'current_value': 'value_B', 'name': 'B'},
            'D': {'current_value': 'value_D', 'name': 'D'},
        }
        interventions = {
            'C': {'current_value': 'value_C', 'name': 'C'}
        }
        blocking_observations = {
            'D': {'current_value': 'value_D', 'name': 'D'}
        }
        pytest.raises(nx.exception.NetworkXError, find_non_blocking_observations, diamond_graph, 'C', observations, blocking_observations, interventions)
