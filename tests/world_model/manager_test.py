
import networkx as nx
import pytest
from typing import Generator

from causal_world_modelling_agent.syntax.definitions import InferredVariableDefinition, CausalRelationshipDefinition
from causal_world_modelling_agent.world_model.world_manager import BaseWorldManager, Query


class TestWorldManager:

    @pytest.fixture
    def empty_world_manager(self):
        return BaseWorldManager()
    
    @pytest.fixture
    def empty_world(self):
        return nx.DiGraph()
    
    @pytest.fixture(scope="module", params=list(range(1, 5)))
    def populated_world(self, request):
        graph = nx.DiGraph()
        graph.add_node("A", **InferredVariableDefinition(
            name="A", 
            description="Node A", 
            type="typeA", 
            values=[f'value_A{i}' for i in range(1, 4)],
            current_value=f'value_A{request.param}',
            causal_effect=f'value_A{request.param}',
            contextual_information="context_A",
            supporting_text_snippets=["snippet1", "snippet2"]
        ).to_dict())
        graph.add_node("B", **InferredVariableDefinition(
            name="B", 
            description="Node B", 
            type="typeB",
            values=[f'value_B{i}' for i in range(1, 4)],
            current_value=f'value_B{request.param}',
            causal_effect=f'value_B{request.param}',
            contextual_information="context_B",
            supporting_text_snippets=["snippet3", "snippet4"]
        ).to_dict())
        graph.add_edge("A", "B", **CausalRelationshipDefinition(
            cause="A", 
            effect="B", 
            description="A causes B",
            contextual_information="causal context",
            type="causal",
        ).to_dict())
        return graph
    
    @pytest.fixture(scope="module", params=list(range(1, 5)))
    def diamond_world(self, request):
        graph = nx.DiGraph()
        graph.add_node("A", **InferredVariableDefinition(
            name="A", 
            description="Node A", 
            type="typeA", 
            values=[f'value_A{i}' for i in range(1, 4)],
            current_value=f'value_A{request.param}',
            causal_effect=f'value_A{request.param}',
            contextual_information="context_A",
            supporting_text_snippets=["snippet1", "snippet2"]
        ).to_dict())
        graph.add_node("B", **InferredVariableDefinition(
            name="B", 
            description="Node B", 
            type="typeB",
            values=[f'value_B{i}' for i in range(1, 4)],
            current_value=f'value_B{request.param}',
            causal_effect=f'value_B{request.param}',
            contextual_information="context_B",
            supporting_text_snippets=["snippet3", "snippet4"]
        ).to_dict())
        graph.add_node("C", **InferredVariableDefinition(
            name="C", 
            description="Node C", 
            type="typeC", 
            values=[f'value_C{i}' for i in range(1, 4)],
            current_value="value_C1",
            causal_effect="value_C1",
            contextual_information="context_C",
            supporting_text_snippets=["snippet5", "snippet6"]
        ).to_dict())
        graph.add_node("D", **InferredVariableDefinition(
            name="D", 
            description="Node D", 
            type="typeD",
            values=[f'value_D{i}' for i in range(1, 4)],
            current_value="value_D1",
            causal_effect="value_D1",
            contextual_information="context_D",
            supporting_text_snippets=["snippet7", "snippet8"]
        ).to_dict())
        graph.add_edge("A", "B", **CausalRelationshipDefinition(
            cause="A", 
            effect="B", 
            description="A causes B",
            contextual_information="causal context",
            type="causal",
        ).to_dict())
        graph.add_edge("A", "C", **CausalRelationshipDefinition(
            cause="A", 
            effect="C", 
            description="A causes C",
            contextual_information="causal context",
            type="causal",
        ).to_dict())
        graph.add_edge("B", "D", **CausalRelationshipDefinition(
            cause="B", 
            effect="D", 
            description="B causes D",
            contextual_information="causal context",
            type="causal",
        ).to_dict())
        graph.add_edge("C", "D", **CausalRelationshipDefinition(
            cause="C", 
            effect="D", 
            description="C causes D",
            contextual_information="causal context",
            type="causal",
        ).to_dict())
        return graph
    
    @pytest.fixture(scope="module", params=list(range(1, 5)))
    def chain_world(self, request):
        graph = nx.DiGraph()
        for i in range(1, 5):
            graph.add_node(f"Node_{i}", **InferredVariableDefinition(
                name=f"Node_{i}", 
                description=f"Node {i}", 
                type="type{i}", 
                values=[f'value_{i}_{j}' for j in range(1, 4)],
                current_value=f'value_{i}_{request.param}',
                causal_effect=f'value_{i}_{request.param}',
                contextual_information=f"context_{i}",
                supporting_text_snippets=[f"snippet{j}" for j in range(1, 3)]
            ).to_dict())
            if i > 1:
                graph.add_edge(f"Node_{i-1}", f"Node_{i}", **CausalRelationshipDefinition(
                    cause=f"Node_{i-1}", 
                    effect=f"Node_{i}", 
                    description=f"Node {i-1} causes Node {i}",
                    contextual_information="causal context",
                    type="causal",
                ).to_dict())
        return graph
    
    @pytest.fixture
    def populated_world_manager(self, populated_world):
        return BaseWorldManager(graphs=[populated_world])
    
    @pytest.fixture
    def diamond_world_manager(self, diamond_world):
        return BaseWorldManager(graphs=[diamond_world])
    
    @pytest.fixture
    def chain_world_manager(self, chain_world):
        return BaseWorldManager(graphs=[chain_world])
    
    @pytest.fixture
    def diamond_world_manager_multiple(self, diamond_world):
        world_0 = diamond_world.copy()

        world_1 = diamond_world.copy()
        world_1.nodes['A']['current_value'] = 'altered_value_A1'
        world_1.nodes['A']['causal_effect'] = 'altered_effect_A1'
        world_1.nodes['C']['current_value'] = 'altered_value_C1'
        world_1.nodes['C']['causal_effect'] = 'altered_effect_C1'
        world_1.nodes['D']['current_value'] = 'altered_value_D1'
        world_1.nodes['D']['causal_effect'] = 'altered_effect_D1'

        return BaseWorldManager(graphs=[world_0, world_1])


    def test_get_graph(self, empty_world_manager):
        graph = empty_world_manager.get_complete_graph()
        assert isinstance(graph, nx.DiGraph)
        assert len(graph.nodes) == 0
        assert len(graph.edges) == 0

    def test_get_traversal_cutoff(self, empty_world_manager):
        cutoff = empty_world_manager.get_traversal_cutoff()
        assert cutoff is None

    def test_set_traversal_cutoff(self, empty_world_manager):
        empty_world_manager.set_traversal_cutoff(5)
        cutoff = empty_world_manager.get_traversal_cutoff()
        assert cutoff == 5
        
    def test_merge(self, empty_world_manager, populated_world):
        empty_world_manager.merge(populated_world)
        graph = empty_world_manager.get_complete_graph()
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert ("A", "B") in graph.edges

        assert graph.nodes["A"] == {
            "name": "A",
            "description": "Node A",
            "type": "typeA",
            "values": [f'value_A{i}' for i in range(1, 4)],
            "world_0": {
                "current_value": populated_world.nodes["A"]["current_value"],
                "causal_effect": populated_world.nodes["A"]["causal_effect"],
                "contextual_information": "context_A",
                "supporting_text_snippets": ["snippet1", "snippet2"]
            }
        }

        assert graph.nodes["B"] == {
            "name": "B",
            "description": "Node B",
            "type": "typeB",
            "values": [f'value_B{i}' for i in range(1, 4)],
            "world_0": {
                "current_value": populated_world.nodes["B"]["current_value"],
                "causal_effect": populated_world.nodes["B"]["causal_effect"],
                "contextual_information": "context_B",
                "supporting_text_snippets": ["snippet3", "snippet4"]
            }
        }

        assert graph.edges["A", "B"]["cause"] == "A"
        assert graph.edges["A", "B"]["effect"] == "B"
        assert graph.edges["A", "B"]["description"] == "A causes B"
        assert graph.edges["A", "B"]["contextual_information"] == "causal context"
        assert graph.edges["A", "B"]["type"] == "causal"

    def test_merge_2(self, empty_world_manager, populated_world):
        empty_world_manager.merge(populated_world)
        graph = empty_world_manager.get_complete_graph()
        
        assert len(graph.nodes) == 2
        assert len(graph.edges) == 1
        
        # Merge again with the same world
        empty_world_manager.merge(populated_world)
        
        graph_after_merge = empty_world_manager.get_complete_graph()
        
        assert graph.nodes == graph_after_merge.nodes
        assert graph.edges == graph_after_merge.edges

        assert graph_after_merge.nodes["A"] == {
            "name": "A",
            "description": "Node A",
            "type": "typeA",
            "values": [f'value_A{i}' for i in range(1, 4)],
            "world_0": {
                "current_value": populated_world.nodes["A"]["current_value"],
                "causal_effect": populated_world.nodes["A"]["causal_effect"],
                "contextual_information": "context_A",
                "supporting_text_snippets": ["snippet1", "snippet2"]
            },
            "world_1": {
                "current_value": populated_world.nodes["A"]["current_value"],
                "causal_effect": populated_world.nodes["A"]["causal_effect"],
                "contextual_information": "context_A",
                "supporting_text_snippets": ["snippet1", "snippet2"]
            }
        }

        assert graph_after_merge.nodes["B"] == {
            "name": "B",
            "description": "Node B",
            "type": "typeB",
            "values": [f'value_B{i}' for i in range(1, 4)],
            "world_0": {
                "current_value": populated_world.nodes["B"]["current_value"],
                "causal_effect": populated_world.nodes["B"]["causal_effect"],
                "contextual_information": "context_B",
                "supporting_text_snippets": ["snippet3", "snippet4"]
            },
            "world_1": {
                "current_value": populated_world.nodes["B"]["current_value"],
                "causal_effect": populated_world.nodes["B"]["causal_effect"],
                "contextual_information": "context_B",
                "supporting_text_snippets": ["snippet3", "snippet4"]
            }
        }

    def test_get_world(self, populated_world_manager):
        world = populated_world_manager.get_world('world_0')
        assert isinstance(world, nx.DiGraph)
        assert len(world.nodes) == 2
        assert len(world.edges) == 1
        
        assert "A" in world.nodes
        assert "B" in world.nodes
        assert ("A", "B") in world.edges

        assert world.nodes["A"] == {
            "name": "A",
            "description": "Node A",
            "type": "typeA",
            "values": [f'value_A{i}' for i in range(1, 4)],
            "current_value": populated_world_manager.get_complete_graph().nodes["A"]["world_0"]["current_value"],
            "causal_effect": populated_world_manager.get_complete_graph().nodes["A"]["world_0"]["causal_effect"],
            "contextual_information": "context_A",
            "supporting_text_snippets": ["snippet1", "snippet2"]
        }

        assert world.nodes["B"] == {
            "name": "B",
            "description": "Node B",
            "type": "typeB",
            "values": [f'value_B{i}' for i in range(1, 4)],
            "current_value": populated_world_manager.get_complete_graph().nodes["B"]["world_0"]["current_value"],
            "causal_effect": populated_world_manager.get_complete_graph().nodes["B"]["world_0"]["causal_effect"],
            "contextual_information": "context_B",
            "supporting_text_snippets": ["snippet3", "snippet4"]
        }

    def test_get_world_invalid(self, populated_world_manager):
        with pytest.raises(ValueError):
            populated_world_manager.get_world('invalid_world')

    def test_get_worlds(self, populated_world_manager):
        worlds = populated_world_manager.get_worlds()
        assert isinstance(worlds, dict)
        assert len(worlds) == 1
        assert 'world_0' in worlds
        assert isinstance(worlds['world_0'], nx.DiGraph)
        assert len(worlds['world_0'].nodes) == 2
        assert len(worlds['world_0'].edges) == 1
        
        assert "A" in worlds['world_0'].nodes
        assert "B" in worlds['world_0'].nodes
        assert ("A", "B") in worlds['world_0'].edges

    def test_get_worlds_empty(self, empty_world_manager):
        worlds = empty_world_manager.get_worlds()
        assert isinstance(worlds, dict)
        assert len(worlds) == 0

    def test_get_worlds_multiple(self, populated_world_manager, populated_world):
        populated_world_manager.merge(populated_world)
        worlds = populated_world_manager.get_worlds()
        assert isinstance(worlds, dict)
        assert len(worlds) == 2
        assert 'world_0' in worlds
        assert 'world_1' in worlds
        assert isinstance(worlds['world_0'], nx.DiGraph)
        assert isinstance(worlds['world_1'], nx.DiGraph)
        assert len(worlds['world_0'].nodes) == 2
        assert len(worlds['world_1'].nodes) == 2
        assert len(worlds['world_0'].edges) == 1
        assert len(worlds['world_1'].edges) == 1
        assert "A" in worlds['world_0'].nodes
        assert "B" in worlds['world_0'].nodes
        assert "A" in worlds['world_1'].nodes
        assert "B" in worlds['world_1'].nodes
        assert ("A", "B") in worlds['world_0'].edges
        assert ("A", "B") in worlds['world_1'].edges
        
    def test_get_worlds_from_node(self, populated_world_manager):
        worlds = populated_world_manager.get_worlds_from_node("A")
        assert isinstance(worlds, dict)
        assert len(worlds) == 1
        assert 'world_0' in worlds
        assert isinstance(worlds['world_0'], nx.DiGraph)
        assert len(worlds['world_0'].nodes) == 2
        assert len(worlds['world_0'].edges) == 1
        
        assert "A" in worlds['world_0'].nodes
        assert "B" in worlds['world_0'].nodes
        assert ("A", "B") in worlds['world_0'].edges

    def test_get_worlds_from_node_disconnected(self, populated_world_manager):
        populated_world_manager.graph.remove_edge("A", "B")
        worlds = populated_world_manager.get_worlds_from_node("A")
        assert isinstance(worlds, dict)
        assert len(worlds) == 1
        assert 'world_0' in worlds
        assert isinstance(worlds['world_0'], nx.DiGraph)
        assert len(worlds['world_0'].nodes) == 1
        assert len(worlds['world_0'].edges) == 0

    def test_get_worlds_from_node_invalid(self, populated_world_manager):
        worlds = populated_world_manager.get_worlds_from_node("invalid_node")
        assert isinstance(worlds, dict)
        assert len(worlds) == 0

    def test_generate_observations(self, populated_world_manager):
        observations = populated_world_manager.generate_observations('B')
        assert isinstance(observations, Generator)
        observations = list(observations)
        assert len(observations) == 1

        assert isinstance(observations[0], Query)
        query = observations[0]

        assert query.is_counterfactual == False
        assert query.world_ids == ["world_0"]
        assert query.target_variable == "B"
        assert list(query.causal_graph.nodes) == list(populated_world_manager.get_complete_graph().nodes)
        assert list(query.causal_graph.edges) == list(populated_world_manager.get_complete_graph().edges)
        assert query.ground_truth == populated_world_manager.get_complete_graph().nodes["B"]["world_0"]["current_value"]
        assert query.is_pseudo_gt == False

        for node, attrs in query.causal_graph.nodes(data=True):
            assert len(attrs) == 4

            assert "current_value" not in attrs
            assert "causal_effect" not in attrs
            assert "contextual_information" not in attrs
            assert "supporting_text_snippets" not in attrs
            
            for attr in attrs:
                assert query.causal_graph.nodes[node][attr] == populated_world_manager.get_complete_graph().nodes[node][attr]

        assert len(query.observations) == 1
        assert query.observations[0] == {
            "name": "A",
            "description": "Node A",
            "type": "typeA",
            "values": [f'value_A{i}' for i in range(1, 4)],
            "current_value": populated_world_manager.get_complete_graph().nodes["A"]["world_0"]["current_value"],
            "causal_effect": populated_world_manager.get_complete_graph().nodes["A"]["world_0"]["current_value"],
            "contextual_information": "context_A",
            "supporting_text_snippets": ["snippet1", "snippet2"]
        }

        assert query.interventions is None

    def test_generate_observations_invalid(self, populated_world_manager):
        observations = populated_world_manager.generate_observations('invalid_node')
        assert isinstance(observations, Generator)
        observations = list(observations)
        assert len(observations) == 0

    def test_generate_observations_multiple_worlds(self, populated_world_manager):
        populated_world_manager.merge(populated_world_manager.get_world('world_0'))
        populated_world_manager.merge(populated_world_manager.get_world('world_0'))
        observations = populated_world_manager.generate_observations('B')
        assert isinstance(observations, Generator)
        observations = list(observations)
        assert len(observations) == 3

        for i, observation in enumerate(observations):
            assert isinstance(observation, Query)
            assert observation.is_counterfactual == False
            assert observation.world_ids == [f"world_{i}"]
            assert observation.target_variable == "B"
            assert list(observation.causal_graph.nodes) == list(populated_world_manager.get_complete_graph().nodes)
            assert list(observation.causal_graph.edges) == list(populated_world_manager.get_complete_graph().edges)
            assert observation.ground_truth == populated_world_manager.get_complete_graph().nodes["B"][f"world_{i}"]["current_value"]
            assert observation.is_pseudo_gt == False

    def test_generate_observations_diamond(self, diamond_world_manager):
        observations = diamond_world_manager.generate_observations('D')
        assert isinstance(observations, Generator)
        observations = list(observations)
        assert len(observations) == 2

        assert isinstance(observations[0], Query)
        query = observations[0]
        assert query.is_counterfactual == False
        assert query.world_ids == ["world_0"]
        assert query.target_variable == "D"
        assert list(query.causal_graph.nodes) == ['B', 'C', 'D']
        assert list(query.causal_graph.edges) == [('B', 'D'), ('C', 'D')]
        assert query.ground_truth == diamond_world_manager.get_complete_graph().nodes["D"]["world_0"]["current_value"]
        assert query.is_pseudo_gt == False
        assert [observation['name'] for observation in query.observations] == ["B", "C"]
        assert query.interventions is None

        assert isinstance(observations[1], Query)
        query = observations[1]
        assert query.is_counterfactual == False
        assert query.world_ids == ["world_0"]
        assert query.target_variable == "D"
        assert list(query.causal_graph.nodes) == list(diamond_world_manager.get_complete_graph().nodes)
        assert list(query.causal_graph.edges) == list(diamond_world_manager.get_complete_graph().edges)
        assert query.ground_truth == diamond_world_manager.get_complete_graph().nodes["D"]["world_0"]["current_value"]
        assert query.is_pseudo_gt == False
        assert [observation['name'] for observation in query.observations] == ["A"]
        assert query.interventions is None

    def test_generate_observations_chain(self, chain_world_manager):
        observations = chain_world_manager.generate_observations('Node_4')
        assert isinstance(observations, Generator)
        observations = list(observations)
        assert len(observations) == 3

        for i, query in enumerate(observations):
            assert isinstance(query, Query)
            assert query.is_counterfactual == False
            assert query.world_ids == ["world_0"]
            assert query.target_variable == "Node_4"
            assert list(query.causal_graph.nodes) == [f"Node_{j}" for j in range(3 - i, 5)]
            assert list(query.causal_graph.edges) == [(f"Node_{j}", f"Node_{j+1}") for j in range(3 - i, 4)]
            assert query.ground_truth == chain_world_manager.get_complete_graph().nodes["Node_4"]["world_0"]["current_value"]
            assert query.is_pseudo_gt == False
            assert [observation['name'] for observation in query.observations] == [f"Node_{3 - i}"]
            assert query.interventions is None

    def test_generate_counterfactuals_match_one_world(self, populated_world_manager):
        counterfactuals = populated_world_manager.generate_counterfactuals_match('B')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 0

    def test_generate_counterfactuals_match_multiple_worlds(self, populated_world_manager):
        populated_world_manager.merge(populated_world_manager.get_world('world_0'))
        populated_world_manager.merge(populated_world_manager.get_world('world_0'))
        counterfactuals = populated_world_manager.generate_counterfactuals_match('B')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 0

    def test_generate_counterfactuals_match_diamond_one_world(self, diamond_world_manager):
        counterfactuals = diamond_world_manager.generate_counterfactuals_match('D')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 0

    def test_generate_counterfactuals_match_diamond_multiple_worlds(self, diamond_world_manager):
        diamond_world_manager.merge(diamond_world_manager.get_world('world_0'))
        diamond_world_manager.merge(diamond_world_manager.get_world('world_0'))
        counterfactuals = diamond_world_manager.generate_counterfactuals_match('D')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 0

    def test_generate_counterfactuals_match_diamond_multiple_worlds_altered(self, diamond_world_manager_multiple):
        counterfactuals = diamond_world_manager_multiple.generate_counterfactuals_match('D')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 2

        assert isinstance(counterfactuals[0], Query)
        query = counterfactuals[0]
        assert query.world_ids == ["world_0", "world_1"]
        assert query.target_variable == "D"
        assert set(query.causal_graph.nodes) == set(['D', 'C', 'B'])
        assert set(query.causal_graph.edges) == set([('B', 'D'), ('C', 'D')])
        assert query.ground_truth == diamond_world_manager_multiple.get_complete_graph().nodes["D"]["world_1"]["current_value"]
        assert query.is_pseudo_gt == False
        assert query.is_counterfactual == True
        assert set([observation['name'] for observation in query.observations]) == set(["C", "D"])
        assert set([observation['current_value'] for observation in query.observations]) == set(["value_C1", "value_D1"])
        assert [intervention['name'] for intervention in query.interventions] == ["C"]
        assert [intervention['current_value'] for intervention in query.interventions] == ["altered_value_C1"]

        assert isinstance(counterfactuals[1], Query)
        query = counterfactuals[1]
        assert query.world_ids == ["world_1", "world_0"]
        assert query.target_variable == "D"
        assert set(query.causal_graph.nodes) == set(['C', 'B', 'D'])
        assert set(query.causal_graph.edges) == set([('B', 'D'), ('C', 'D')])
        assert query.ground_truth == diamond_world_manager_multiple.get_complete_graph().nodes["D"]["world_0"]["current_value"]
        assert query.is_pseudo_gt == False
        assert query.is_counterfactual == True
        assert set([observation['name'] for observation in query.observations]) == set(["C", "D"])
        assert set([observation['current_value'] for observation in query.observations]) == set(["altered_value_C1", "altered_value_D1"])
        assert [intervention['name'] for intervention in query.interventions] == ["C"]
        assert [intervention['current_value'] for intervention in query.interventions] == ["value_C1"]

    def test_generate_counterfactuals_mix_linear_one_world(self, populated_world_manager):
        counterfactuals = populated_world_manager.generate_counterfactuals_mix('B')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 0

    def test_generate_counterfactuals_mix_linear_multiple_worlds(self, populated_world_manager):
        populated_world_manager.merge(populated_world_manager.get_world('world_0'))
        populated_world_manager.merge(populated_world_manager.get_world('world_0'))
        counterfactuals = populated_world_manager.generate_counterfactuals_mix('B')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 0

    def test_generate_counterfactuals_mix_linear_diamond_multiple_worlds_altered(self, diamond_world_manager_multiple):
        counterfactuals = diamond_world_manager_multiple.generate_counterfactuals_mix('D')
        assert isinstance(counterfactuals, Generator)
        counterfactuals = list(counterfactuals)
        assert len(counterfactuals) == 2

        assert isinstance(counterfactuals[0], Query)
        query = counterfactuals[0]
        assert query.world_ids == ["world_0", "world_1"]
        assert query.target_variable == "D"
        assert set(query.causal_graph.nodes) == set(['D', 'C', 'B'])
        assert set(query.causal_graph.edges) == set([('B', 'D'), ('C', 'D')])
        assert query.is_pseudo_gt == True
        assert query.is_counterfactual == True
        assert set([observation['name'] for observation in query.observations]) == set(["C", "D"])
        assert set([observation['current_value'] for observation in query.observations]) == set(["value_C1", "value_D1"])
        assert [intervention['name'] for intervention in query.interventions] == ["C"]
        assert [intervention['current_value'] for intervention in query.interventions] == ["altered_value_C1"]

        assert isinstance(query.ground_truth, dict)
        assert len(query.ground_truth) == 2
        assert set(query.ground_truth.keys()) == set(["value_D1", "altered_value_D1"])
        assert abs(query.ground_truth["value_D1"] - 0.9820) < 0.0001
        assert abs(query.ground_truth["altered_value_D1"] - 0.0180) < 0.0001

        assert isinstance(counterfactuals[1], Query)
        query = counterfactuals[1]
        assert query.world_ids == ["world_1", "world_0"]
        assert query.target_variable == "D"
        assert set(query.causal_graph.nodes) == set(['C', 'B', 'D'])
        assert set(query.causal_graph.edges) == set([('B', 'D'), ('C', 'D')])
        assert query.is_pseudo_gt == True
        assert query.is_counterfactual == True
        assert set([observation['name'] for observation in query.observations]) == set(["C", "D"])
        assert set([observation['current_value'] for observation in query.observations]) == set(["altered_value_C1", "altered_value_D1"])
        assert [intervention['name'] for intervention in query.interventions] == ["C"]
        assert [intervention['current_value'] for intervention in query.interventions] == ["value_C1"]

        assert isinstance(query.ground_truth, dict)
        assert len(query.ground_truth) == 2
        assert set(query.ground_truth.keys()) == set(["altered_value_D1", "value_D1"])
        assert abs(query.ground_truth["altered_value_D1"] - 0.9820) < 0.0001
        assert abs(query.ground_truth["value_D1"] - 0.0180) < 0.0001

    def test_generate_counterfactuals_mix_intervals_diamond_multiple_worlds_altered(self, diamond_world_manager_multiple):
        counterfactuals = diamond_world_manager_multiple.generate_counterfactuals_mix('D', mixing_function="intervals")
        counterfactuals = list(counterfactuals)

        query = counterfactuals[0]

        assert isinstance(query.ground_truth, dict)
        assert len(query.ground_truth) == 2
        assert set(query.ground_truth.keys()) == set(["value_D1", "altered_value_D1"])
        assert abs(query.ground_truth["value_D1"] - 0.9820) < 0.0001
        assert abs(query.ground_truth["altered_value_D1"] - 0.0180) < 0.0001

        query = counterfactuals[1]

        assert isinstance(query.ground_truth, dict)
        assert len(query.ground_truth) == 2
        assert set(query.ground_truth.keys()) == set(["altered_value_D1", "value_D1"])
        assert abs(query.ground_truth["altered_value_D1"] - 0.9820) < 0.0001
        assert abs(query.ground_truth["value_D1"] - 0.0180) < 0.0001

    def test_generate_counterfactuals_mix_overlapping_beams_diamond_multiple_worlds_altered(self, diamond_world_manager_multiple):
        counterfactuals = diamond_world_manager_multiple.generate_counterfactuals_mix('D', mixing_function="overlapping_beams")
        counterfactuals = list(counterfactuals)

        query = counterfactuals[0]

        assert isinstance(query.ground_truth, dict)
        assert len(query.ground_truth) == 2
        assert set(query.ground_truth.keys()) == set(["value_D1", "altered_value_D1"])
        assert abs(query.ground_truth["value_D1"] - 0.6667) < 0.0001
        assert abs(query.ground_truth["altered_value_D1"] - 0.3333) < 0.0001

        query = counterfactuals[1]

        assert isinstance(query.ground_truth, dict)
        assert len(query.ground_truth) == 2
        assert set(query.ground_truth.keys()) == set(["altered_value_D1", "value_D1"])
        assert abs(query.ground_truth["altered_value_D1"] - 0.6667) < 0.0001
        assert abs(query.ground_truth["value_D1"] - 0.3333) < 0.0001