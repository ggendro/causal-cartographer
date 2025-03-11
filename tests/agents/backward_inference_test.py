
import networkx as nx
import pytest

from mocks.mock_models import CountInferenceMockModel
from causal_world_modelling_agent.agents.causal_inference.backward_inference_agent import BackwardInferenceAgentFactory




class TestBackwardInferenceAgent:

    @pytest.fixture
    def backward_inference_agent(self):
        return BackwardInferenceAgentFactory().createAgent(CountInferenceMockModel())
    
    @pytest.fixture
    def causal_graph_1(self):
        graph = nx.DiGraph()
        graph.add_node("A", name="A")
        graph.add_node("B", name="B")
        graph.add_node("C", name="C")
        graph.add_node("D", name="D")
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        graph.add_edge("C", "D")
        return graph
    
    @pytest.fixture
    def causal_graph_2(self):
        graph = nx.DiGraph()
        graph.add_node("A", name="A")
        graph.add_node("B", name="B")
        graph.add_node("C", name="C")
        graph.add_node("D", name="D")
        graph.add_node("E", name="E")
        graph.add_node("F", name="F")
        graph.add_node("G", name="G")
        graph.add_node("H", name="H")
        graph.add_edge("A", "B")
        graph.add_edge("A", "C")
        graph.add_edge("B", "D")
        graph.add_edge("C", "D")
        graph.add_edge("D", "E")
        graph.add_edge("E", "F")
        graph.add_edge("E", "G")
        graph.add_edge("F", "H")
        graph.add_edge("G", "H")
        return graph
    
    def test_run_1_no_observations(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = [{
            "name": "D",
            "current_value": 1
        }]
        observations = []
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert 'A' not in answer_graph
        assert 'B' not in answer_graph
        assert 'C' not in answer_graph
        assert answer_graph.nodes['D']['causal_effect'] == 1

    def test_run_1_with_observations(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = [{
            "name": "D",
            "current_value": 1
        }]
        observations = [{
            "name": "A",
            "current_value": 0
        }]
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert answer_graph.nodes['A']['causal_effect'] == 1
        assert answer_graph.nodes['B']['causal_effect'] == 0
        assert answer_graph.nodes['C']['causal_effect'] == 0
        assert answer_graph.nodes['D']['causal_effect'] == 1

    def test_run_1_with_observations_2(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = [{
            "name": "D",
            "current_value": 2
        }]
        observations = [{
            "name": "A",
            "current_value": 0
        }, {
            "name": "C",
            "current_value": 1
        }]
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert answer_graph.nodes['A']['causal_effect'] == 3
        assert answer_graph.nodes['B']['causal_effect'] == -1
        assert answer_graph.nodes['C']['causal_effect'] == -1
        assert answer_graph.nodes['D']['causal_effect'] == 2

    def test_run_1_with_observations_3(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = [{
            "name": "D",
            "current_value": 2
        }]
        observations = [{
            "name": "C",
            "current_value": 1
        }]
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert answer_graph.nodes['A']['causal_effect'] == 3
        assert answer_graph.nodes['B']['causal_effect'] == -1
        assert answer_graph.nodes['C']['causal_effect'] == -1
        assert answer_graph.nodes['D']['causal_effect'] == 2

    def test_run_1_with_observations_4(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = [{
            "name": "D",
            "current_value": 3
        }]
        observations = [{
            "name": "A",
            "current_value": 1
            }, {
            "name": "C",
            "current_value": 1
            }, {
            "name": "B",
            "current_value": 0
        }]
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert 'A' not in answer_graph
        assert answer_graph.nodes['B']['causal_effect'] == -2
        assert answer_graph.nodes['C']['causal_effect'] == -2
        assert answer_graph.nodes['D']['causal_effect'] == 3

    def test_run_1_with_observations_5(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = [{
            "name": "D",
            "current_value": 3
        }]
        observations = [{
            "name": "C",
            "current_value": 1
            }, {
            "name": "B",
            "current_value": 0
        }]
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert 'A' not in answer_graph
        assert answer_graph.nodes['B']['causal_effect'] == -2
        assert answer_graph.nodes['C']['causal_effect'] == -2
        assert answer_graph.nodes['D']['causal_effect'] == 3

    def test_run_1_observed_outcome(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = [{
            "name": "D",
            "current_value": 1
        }]
        observations = [{
            "name": "D",
            "current_value": 0
        }]
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        pytest.raises(ValueError, backward_inference_agent.run, "Hello world!", additional_args=additional_args)

    def test_run_1_no_outcomes(self, backward_inference_agent, causal_graph_1):
        observations = []
        additional_args = {
            "causal_graph": causal_graph_1,
            "observations": observations
        }
        pytest.raises(ValueError, backward_inference_agent.run, "Hello world!", additional_args=additional_args)

    def test_run_1_empty_outcomes(self, backward_inference_agent, causal_graph_1):
        counterfactual_outcomes = []
        observations = []
        additional_args = {
            "causal_graph": causal_graph_1,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        pytest.raises(ValueError, backward_inference_agent.run, "Hello world!", additional_args=additional_args)

    def test_run_2_with_observations(self, backward_inference_agent, causal_graph_2):
        counterfactual_outcomes = [{
            "name": "H",
            "current_value": 1
        }]
        observations = [{
            "name": "A",
            "current_value": 0
        }]
        additional_args = {
            "causal_graph": causal_graph_2,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert answer_graph.nodes['A']['causal_effect'] == -1
        assert answer_graph.nodes['B']['causal_effect'] == 1
        assert answer_graph.nodes['C']['causal_effect'] == 1
        assert answer_graph.nodes['D']['causal_effect'] == 0
        assert answer_graph.nodes['E']['causal_effect'] == 1
        assert answer_graph.nodes['F']['causal_effect'] == 0
        assert answer_graph.nodes['G']['causal_effect'] == 0
        assert answer_graph.nodes['H']['causal_effect'] == 1

    def test_run_2_with_observations_2(self, backward_inference_agent, causal_graph_2):
        counterfactual_outcomes = [{
            "name": "H",
            "current_value": 1
        }]
        observations = [{
            "name": "A",
            "current_value": 0
        }, {
            "name": "C",
            "current_value": 1
        }, {
            "name": "E",
            "current_value": 1
        }]
        additional_args = {
            "causal_graph": causal_graph_2,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert 'A' not in answer_graph
        assert 'B' not in answer_graph
        assert 'C' not in answer_graph
        assert 'D' not in answer_graph
        assert answer_graph.nodes['E']['causal_effect'] == 1
        assert answer_graph.nodes['F']['causal_effect'] == 0
        assert answer_graph.nodes['G']['causal_effect'] == 0
        assert answer_graph.nodes['H']['causal_effect'] == 1

    def test_run_2_with_observations_intermediate(self, backward_inference_agent, causal_graph_2):
        counterfactual_outcomes = [{
            "name": "E",
            "current_value": 1
        }]
        observations = [{
            "name": "A",
            "current_value": 0
        }, {
            "name": "C",
            "current_value": 1
        }, {
            "name": "D",
            "current_value": 1
        }]
        additional_args = {
            "causal_graph": causal_graph_2,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert 'A' not in answer_graph
        assert 'B' not in answer_graph
        assert 'C' not in answer_graph
        assert answer_graph.nodes['D']['causal_effect'] == 0
        assert answer_graph.nodes['E']['causal_effect'] == 1
        assert 'F' not in answer_graph
        assert 'G' not in answer_graph
        assert 'H' not in answer_graph

    def test_run_2_with_observations_intermediate_2(self, backward_inference_agent, causal_graph_2):
        counterfactual_outcomes = [{
            "name": "F",
            "current_value": 1
        }]
        observations = [{
            "name": "A",
            "current_value": 0
        }, {
            "name": "C",
            "current_value": 1
        }, {
            "name": "D",
            "current_value": 1
        }]
        additional_args = {
            "causal_graph": causal_graph_2,
            "counterfactual_outcomes": counterfactual_outcomes,
            "observations": observations
        }
        answer_graph = backward_inference_agent.run("Hello world!", additional_args=additional_args)
        assert 'A' not in answer_graph
        assert 'B' not in answer_graph
        assert 'C' not in answer_graph
        assert answer_graph.nodes['D']['causal_effect'] == 1
        assert answer_graph.nodes['E']['causal_effect'] == 0
        assert answer_graph.nodes['F']['causal_effect'] == 1
        assert 'G' not in answer_graph
        assert 'H' not in answer_graph