
import networkx as nx
import pytest

from mocks.mock_models import CountInferenceMockModel
from causal_world_modelling_agent.agents.causal_inference.causal_inference_agent import CausalInferenceAgentFactory




class TestCausalInferenceAgent:

    @pytest.fixture
    def causal_inference_agent(self):
        return CausalInferenceAgentFactory().createAgent(CountInferenceMockModel())
    
    @pytest.fixture
    def causal_graph_1(self):
        graph = nx.DiGraph()
        graph.add_node('A', name='A')
        graph.add_node('B', name='B')
        graph.add_edge('A', 'B')
        return graph
    
    @pytest.fixture
    def causal_graph_2(self):
        graph = nx.DiGraph()
        graph.add_node('A', name='A')
        graph.add_node('B', name='B')
        graph.add_node('C', name='C')
        graph.add_node('D', name='D')
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'D')
        graph.add_edge('C', 'D')
        return graph
    
    @pytest.fixture
    def causal_graph_3(self):
        graph = nx.DiGraph()
        graph.add_node('A', name='A')
        graph.add_node('B', name='B')
        graph.add_node('C', name='C')
        graph.add_node('D', name='D')
        graph.add_node('E', name='E')
        graph.add_node('F', name='F')
        graph.add_node('G', name='G')
        graph.add_edge('A', 'B')
        graph.add_edge('A', 'C')
        graph.add_edge('B', 'D')
        graph.add_edge('C', 'D')
        graph.add_edge('D', 'E')
        graph.add_edge('D', 'F')
        graph.add_edge('E', 'G')
        graph.add_edge('F', 'G')
        graph.add_edge('A', 'G')
        return graph
    
    @pytest.fixture
    def causal_graph_4(self):
        graph = nx.DiGraph()
        graph.add_node('A', name='A')
        graph.add_node('B', name='B')
        graph.add_node('C', name='C')
        graph.add_edge('A', 'B')
        graph.add_edge('B', 'C')
        return graph
    
    @pytest.fixture
    def causal_graph_5(self):
        graph = nx.DiGraph()
        graph.add_node('A', name='A')
        graph.add_node('B', name='B')
        graph.add_node('C', name='C')
        graph.add_edge('B', 'A')
        graph.add_edge('B', 'C')
        return graph
    
    def test_run_1_no_observations(self, causal_inference_agent, causal_graph_1):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_1, 'target_variable': 'B'})
        
        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == 1

    def test_run_1_intermediate(self, causal_inference_agent, causal_graph_1):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_1, 'target_variable': 'A'})
        
        assert answer_text == 1
        assert 'B' not in answer_graph.nodes
        assert answer_graph.nodes['A']['causal_effect'] == 1

    def test_run_1_with_observations(self, causal_inference_agent, causal_graph_1):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_1, 'target_variable': 'B', 'observations': [{'name': 'A', 'current_value': 2}]})
        
        assert answer_text == 3
        assert answer_graph.nodes['A']['causal_effect'] == 2
        assert answer_graph.nodes['B']['causal_effect'] == 3

    def test_run_1_target_observed(self, causal_inference_agent, causal_graph_1):
        pytest.raises(ValueError, causal_inference_agent.run, "Hello world!", additional_args={'causal_graph': causal_graph_1, 'target_variable': 'A', 'observations': [{'name': 'A', 'current_value': 2}]})

    def test_run_1_target_intervened(self, causal_inference_agent, causal_graph_1):
        pytest.raises(ValueError, causal_inference_agent.run, "Hello world!", additional_args={'causal_graph': causal_graph_1, 'target_variable': 'A', 'interventions': [{'name': 'A', 'current_value': 2}]})

    def test_run_2_no_observations(self, causal_inference_agent, causal_graph_2):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_2, 'target_variable': 'D'})
        
        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert 'B' not in answer_graph.nodes
        assert 'C' not in answer_graph.nodes
        assert answer_graph.nodes['D']['causal_effect'] == 1

    def test_run_2_no_observations_intermediate(self, causal_inference_agent, causal_graph_2):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_2, 'target_variable': 'C'})
        
        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert 'B' not in answer_graph.nodes
        assert answer_graph.nodes['C']['causal_effect'] == 1
        assert 'D' not in answer_graph.nodes

    def test_run_2_with_observations(self, causal_inference_agent, causal_graph_2):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_2, 'target_variable': 'D', 'observations': [{'name': 'A', 'current_value': 2}]})
        
        assert answer_text == 7
        assert answer_graph.nodes['A']['causal_effect'] == 2
        assert answer_graph.nodes['B']['causal_effect'] == 3
        assert answer_graph.nodes['C']['causal_effect'] == 3
        assert answer_graph.nodes['D']['causal_effect'] == 7

    def test_run_2_with_observations_intermediate(self, causal_inference_agent, causal_graph_2):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_2, 'target_variable': 'C', 'observations': [{'name': 'A', 'current_value': 2}]})
        
        assert answer_text == 3
        assert answer_graph.nodes['A']['causal_effect'] == 2
        assert 'B' not in answer_graph.nodes
        assert answer_graph.nodes['C']['causal_effect'] == 3
        assert 'D' not in answer_graph.nodes

    def test_run_2_with_interventions(self, causal_inference_agent, causal_graph_2):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_2, 'target_variable': 'D', 'observations': [{'name': 'A', 'current_value': 2}], 'interventions': [{'name': 'C', 'current_value': 0}]})
        
        assert answer_text == 4
        assert answer_graph.nodes['A']['causal_effect'] == 2
        assert answer_graph.nodes['B']['causal_effect'] == 3
        assert answer_graph.nodes['C']['causal_effect'] == 0
        assert answer_graph.nodes['D']['causal_effect'] == 4

    def test_run_2_with_abductions(self, causal_inference_agent, causal_graph_2):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_2, 'target_variable': 'D', 'observations': [{'name': 'C', 'current_value': 2}], 'interventions': [{'name': 'C', 'current_value': 0}]})
        
        assert answer_text == 3
        assert answer_graph.nodes['A']['causal_effect'] == 1
        assert answer_graph.nodes['B']['causal_effect'] == 2
        assert answer_graph.nodes['C']['causal_effect'] == 0
        assert answer_graph.nodes['D']['causal_effect'] == 3

    def test_run_3_with_observations(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'target_variable': 'G', 'observations': [{'name': 'A', 'current_value': 1}]})
        
        assert answer_text == 14
        assert answer_graph.nodes['A']['causal_effect'] == 1
        assert answer_graph.nodes['B']['causal_effect'] == 2
        assert answer_graph.nodes['C']['causal_effect'] == 2
        assert answer_graph.nodes['D']['causal_effect'] == 5
        assert answer_graph.nodes['E']['causal_effect'] == 6
        assert answer_graph.nodes['F']['causal_effect'] == 6
        assert answer_graph.nodes['G']['causal_effect'] == 14

    def test_run_3_with_observations_2(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'target_variable': 'G', 'observations': [{'name': 'A', 'current_value': 1}, {'name': 'D', 'current_value': 1}]})
        
        assert answer_text == 6
        assert answer_graph.nodes['A']['causal_effect'] == 1
        assert 'B' not in answer_graph.nodes
        assert 'C' not in answer_graph.nodes
        assert answer_graph.nodes['D']['causal_effect'] == 1
        assert answer_graph.nodes['E']['causal_effect'] == 2
        assert answer_graph.nodes['F']['causal_effect'] == 2
        assert answer_graph.nodes['G']['causal_effect'] == 6

    def test_run_3_with_interventions(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'target_variable': 'G', 'observations': [{'name': 'A', 'current_value': 1}], 'interventions': [{'name': 'B', 'current_value': 4}, {'name': 'F', 'current_value': 4}]})
        
        assert answer_text == 14
        assert answer_graph.nodes['A']['causal_effect'] == 1
        assert answer_graph.nodes['B']['causal_effect'] == 4
        assert answer_graph.nodes['C']['causal_effect'] == 2
        assert answer_graph.nodes['D']['causal_effect'] == 7
        assert answer_graph.nodes['E']['causal_effect'] == 8
        assert answer_graph.nodes['F']['causal_effect'] == 4
        assert answer_graph.nodes['G']['causal_effect'] == 14

    def test_run_3_with_interventions_2(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'target_variable': 'G', 'interventions': [{'name': 'B', 'current_value': 4}, {'name': 'F', 'current_value': 4}]})

        assert answer_text == 11
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == 4
        assert 'C' not in answer_graph.nodes
        assert answer_graph.nodes['D']['causal_effect'] == 5
        assert answer_graph.nodes['E']['causal_effect'] == 6
        assert answer_graph.nodes['F']['causal_effect'] == 4
        assert answer_graph.nodes['G']['causal_effect'] == 11

    def test_run_3_with_abductions(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'target_variable': 'G', 'observations': [{'name': 'B', 'current_value': 2}, {'name': 'F', 'current_value': 2}], 'interventions': [{'name': 'B', 'current_value': 4}, {'name': 'F', 'current_value': 4}]})

        assert answer_text == 0
        assert answer_graph.nodes['A']['causal_effect'] == -6
        assert answer_graph.nodes['B']['causal_effect'] == 4
        assert answer_graph.nodes['C']['causal_effect'] == -5
        assert answer_graph.nodes['D']['causal_effect'] == 0
        assert answer_graph.nodes['E']['causal_effect'] == 1
        assert answer_graph.nodes['F']['causal_effect'] == 4
        assert answer_graph.nodes['G']['causal_effect'] == 0

    def test_run_3_with_abductions_2(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'target_variable': 'G', 'observations': [{'name': 'B', 'current_value': 2}, {'name': 'F', 'current_value': 2}], 'interventions': [{'name': 'C', 'current_value': 4}, {'name': 'E', 'current_value': 4}]})

        assert answer_text == 3
        assert answer_graph.nodes['A']['causal_effect'] == -4
        assert 'causal_effect' not in answer_graph.nodes['B']
        assert 'C' not in answer_graph.nodes
        assert 'D' not in answer_graph.nodes
        assert answer_graph.nodes['E']['causal_effect'] == 4
        assert answer_graph.nodes['F']['causal_effect'] == 2
        assert answer_graph.nodes['G']['causal_effect'] == 3

    def test_run_4_no_observations(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C'})
        
        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert 'B' not in answer_graph.nodes
        assert answer_graph.nodes['C']['causal_effect'] == 1

    def test_run_4_with_observations(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'observations': [{'name': 'A', 'current_value': 2}]})
        
        assert answer_text == 4
        assert answer_graph.nodes['A']['causal_effect'] == 2
        assert answer_graph.nodes['B']['causal_effect'] == 3
        assert answer_graph.nodes['C']['causal_effect'] == 4

    def test_run_4_with_observations_2(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'observations': [{'name': 'B', 'current_value': 4}]})

        assert answer_text == 5
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == 4
        assert answer_graph.nodes['C']['causal_effect'] == 5

    def test_run_4_with_observations_3(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'A', 'observations': [{'name': 'B', 'current_value': 2}, {'name': 'C', 'current_value': 4}]})

        assert answer_text == 1
        assert answer_graph.nodes['A']['causal_effect'] == 1
        assert 'B' not in answer_graph.nodes
        assert 'C' not in answer_graph.nodes

    def test_run_4_with_interventions(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'interventions': [{'name': 'A', 'current_value': 0}]})

        assert answer_text == 2
        assert answer_graph.nodes['A']['causal_effect'] == 0
        assert answer_graph.nodes['B']['causal_effect'] == 1
        assert answer_graph.nodes['C']['causal_effect'] == 2

    def test_run_4_with_interventions_2(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'interventions': [{'name': 'B', 'current_value': 4}]})

        assert answer_text == 5
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == 4
        assert answer_graph.nodes['C']['causal_effect'] == 5

    def test_run_4_with_interventions_3(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'interventions': [{'name': 'A', 'current_value': 0}, {'name': 'B', 'current_value': 4}]})

        assert answer_text == 5
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == 4
        assert answer_graph.nodes['C']['causal_effect'] == 5

    def test_run_4_with_interventions_4(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'observations': [{'name': 'A', 'current_value': 0}], 'interventions': [{'name': 'B', 'current_value': -1}]})

        assert answer_text == 0
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == -1
        assert answer_graph.nodes['C']['causal_effect'] == 0

    def test_run_4_with_intervention_5(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'observations': [{'name': 'B', 'current_value': 0}], 'interventions': [{'name': 'A', 'current_value': -1}]})

        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == 0
        assert answer_graph.nodes['C']['causal_effect'] == 1

    def test_run_5_with_observations(self, causal_inference_agent, causal_graph_5):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_5, 'target_variable': 'C', 'observations': [{'name': 'A', 'current_value': 2}]})

        assert answer_text == 0
        assert 'causal_effect' not in answer_graph.nodes['A']
        assert answer_graph.nodes['B']['causal_effect'] == -1
        assert answer_graph.nodes['C']['causal_effect'] == 0

    def test_run_5_with_interventions(self, causal_inference_agent, causal_graph_5):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_5, 'target_variable': 'C', 'interventions': [{'name': 'A', 'current_value': 2}]})

        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert 'B' not in answer_graph.nodes
        assert answer_graph.nodes['C']['causal_effect'] == 1
