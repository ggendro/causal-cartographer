
import networkx as nx
import pytest

from mocks.mock_models import CountInferenceMockModel
from causal_world_modelling_agent.agents.causal_inference.causal_inference_agent import CausalInferenceAgentFactory




class TestCausalInferenceAgent:

    @pytest.fixture
    def causal_inference_agent(self):
        agent = CausalInferenceAgentFactory().createAgent(CountInferenceMockModel())
        agent.final_answer_checks = []
        return agent
    
    @pytest.fixture
    def causal_inference_agent_with_checks(self):
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
    
    @pytest.fixture
    def causal_graph_full_features(self):
        graph = nx.DiGraph()
        graph.add_nodes_from([
            ("Price of Oranges", {
            "name": "Price of Oranges",
            "description": "The price of oranges in the market, which is influenced by supply and demand.",
            "type": "float",
            "values": "range(0, 10)",
            "causal_effect": "The price of oranges depends on the quantity demanded and quantity supplied in the market.",
            "current_value": "2.50",
            "contextual_information": "The current price of oranges is $2.50 per kilogram, based on current market conditions."
            }),
            ("Quantity Demanded", {
            "name": "Quantity Demanded",
            "description": "The total number of kilograms of oranges that consumers want to buy at a given price.",
            "type": "integer",
            "values": "range(0, 1000)",
            "causal_effect": "The quantity demanded generally increases as the price decreases, according to the law of demand.",
            "current_value": "600",
            "contextual_information": "At the observed price of $2.50 per kilogram, consumers demand 600 kilograms of oranges."
            }),
            ("Quantity Supplied", {
            "name": "Quantity Supplied",
            "description": "The total number of kilograms of oranges that producers are willing to sell at a given price.",
            "type": "integer",
            "values": "range(0, 1000)",
            "causal_effect": "The quantity supplied generally increases as the price increases, according to the law of supply.",
            "current_value": "500",
            "contextual_information": "At the observed price of $2.50 per kilogram, producers are willing to supply 500 kilograms of oranges."
            }),
            ("Market Constant", {
            "name": "Market Constant",
            "description": "A constant factor that adjusts the relationship between supply, demand, and price in the market.",
            "type": "float",
            "values": "[1.0, 1.5, 2.0]",
            "causal_effect": "The market constant adjusts how supply and demand influence the final price of oranges.",
            "current_value": "1.2",
            "contextual_information": "The market constant is set to 1.2, based on the current market conditions for oranges."
            })
        ])
        graph.add_edges_from([
            ("Quantity Demanded", "Price of Oranges", {
                "cause": "Quantity Demanded",
                "effect": "Price of Oranges",
                "description": "The price of oranges is influenced by the quantity demanded and the quantity supplied in the market. As demand increases, price tends to rise, and as demand decreases, price tends to fall.",
                "contextual_information": "With 600 kilograms of oranges demanded at the current price of $2.50, the price is expected to change if the demand increases or decreases.",
                "type": "direct",
                "strength": "high",
                "confidence": "high",
                "function": "lambda market_constant, demand, supply: (market_constant * demand) / supply if supply != 0 else float('inf')"
            }),
            ("Quantity Supplied", "Price of Oranges", {
                "cause": "Quantity Supplied",
                "effect": "Price of Oranges",
                "description": "The price of oranges is influenced by the quantity supplied. If supply exceeds demand, prices tend to fall, and if supply falls short of demand, prices tend to rise.",
                "contextual_information": "Currently, the supply is lower than demand (500 kilograms supplied vs. 600 kilograms demanded), suggesting upward pressure on the price.",
                "type": "direct",
                "strength": "high",
                "confidence": "high",
                "function": "lambda market_constant, demand, supply: (market_constant * demand) / supply if supply != 0 else float('inf')"
            }),
            ("Market Constant", "Price of Oranges", {
                "cause": "Market Constant",
                "effect": "Price of Oranges",
                "description": "The market constant adjusts the relationship between supply, demand, and price in the market. A higher constant leads to higher prices, while a lower constant leads to lower prices.",
                "contextual_information": "With a market constant of 1.2, the price of oranges is expected to be 1.2 times the ratio of demand to supply.",
                "type": "direct",
                "strength": "moderate",
                "confidence": "medium",
                "function": "lambda market_constant, demand, supply: (market_constant * demand) / supply if supply != 0 else float('inf')"
            })
        ])
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

    def test_run_2_with_abductions_is_counterfactual(self, causal_inference_agent, causal_graph_2):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_2, 'is_counterfactual': True, 'target_variable': 'D', 'observations': [{'name': 'C', 'current_value': 2}], 'interventions': [{'name': 'C', 'current_value': 0}]})
        
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

    def test_run_3_with_abductions_is_counterfactual(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'is_counterfactual': True, 'target_variable': 'G', 'observations': [{'name': 'B', 'current_value': 2}, {'name': 'F', 'current_value': 2}], 'interventions': [{'name': 'B', 'current_value': 4}, {'name': 'F', 'current_value': 4}]})

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

    def test_run_3_with_abductions_2_is_counterfactual(self, causal_inference_agent, causal_graph_3):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_3, 'is_counterfactual': True, 'target_variable': 'G', 'observations': [{'name': 'B', 'current_value': 2}, {'name': 'F', 'current_value': 2}], 'interventions': [{'name': 'C', 'current_value': 4}, {'name': 'E', 'current_value': 4}]})

        assert answer_text == 0
        assert answer_graph.nodes['A']['causal_effect'] == -6
        assert answer_graph.nodes['B']['causal_effect'] == -5
        assert answer_graph.nodes['C']['causal_effect'] == 4
        assert answer_graph.nodes['D']['causal_effect'] == 0
        assert answer_graph.nodes['E']['causal_effect'] == 4
        assert answer_graph.nodes['F']['causal_effect'] == 1
        assert answer_graph.nodes['G']['causal_effect'] == 0

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

    def test_run_4_with_interventions_4_is_counterfactual(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'is_counterfactual': True, 'target_variable': 'C', 'observations': [{'name': 'A', 'current_value': 0}], 'interventions': [{'name': 'B', 'current_value': -1}]})

        assert answer_text == 0
        assert answer_graph.nodes['A']['causal_effect'] == 0
        assert answer_graph.nodes['B']['causal_effect'] == -1
        assert answer_graph.nodes['C']['causal_effect'] == 0

    def test_run_4_with_intervention_5(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'target_variable': 'C', 'observations': [{'name': 'B', 'current_value': 0}], 'interventions': [{'name': 'A', 'current_value': -1}]})

        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert answer_graph.nodes['B']['causal_effect'] == 0
        assert answer_graph.nodes['C']['causal_effect'] == 1

    def test_run_4_with_intervention_5_is_counterfactual(self, causal_inference_agent, causal_graph_4):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_4, 'is_counterfactual': True, 'target_variable': 'C', 'observations': [{'name': 'B', 'current_value': 0}], 'interventions': [{'name': 'A', 'current_value': -2}]})

        assert answer_text == 0
        assert answer_graph.nodes['A']['causal_effect'] == -2
        assert answer_graph.nodes['B']['causal_effect'] == -1
        assert answer_graph.nodes['C']['causal_effect'] == 0

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

    def test_run_5_with_interventions_is_counterfactual(self, causal_inference_agent, causal_graph_5):
        answer_text, answer_graph = causal_inference_agent.run("Hello world!", additional_args={'causal_graph': causal_graph_5, 'is_counterfactual': True, 'target_variable': 'C', 'interventions': [{'name': 'A', 'current_value': 2}]})

        assert answer_text == 1
        assert 'A' not in answer_graph.nodes
        assert 'B' not in answer_graph.nodes
        assert answer_graph.nodes['C']['causal_effect'] == 1

    def test_run_with_features_and_checks(self, causal_inference_agent_with_checks, causal_graph_full_features):
        answer_text, answer_graph = causal_inference_agent_with_checks.run("Hello world!", 
                                                                           additional_args={
                                                                               'causal_graph': causal_graph_full_features, 
                                                                               'target_variable': 'Price of Oranges',
                                                                                 'observations': [
                                                                                      {'name': 'Quantity Demanded', 'current_value': 600},
                                                                                      {'name': 'Quantity Supplied', 'current_value': 500},
                                                                                      {'name': 'Market Constant', 'current_value': 1.2}
                                                                                 ]
                                                                               })
        
        assert answer_text == 1102.2
        assert answer_graph.nodes['Price of Oranges']['causal_effect'] == 1102.2
        assert answer_graph.nodes['Quantity Demanded']['causal_effect'] == 600
        assert answer_graph.nodes['Quantity Supplied']['causal_effect'] == 500
        assert answer_graph.nodes['Market Constant']['causal_effect'] == 1.2
