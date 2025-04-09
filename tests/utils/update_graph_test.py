
import networkx as nx
import pytest

from causal_world_modelling_agent.syntax.definitions import VariableDefinition, InferredVariableDefinition
from causal_world_modelling_agent.utils.inference_utils import update_graph_node_values_in_place, build_intervened_graph


class TestUpdateGraphNodeValuesInPlace:
    
    @pytest.fixture
    def graph(self):
        G = nx.DiGraph()
        G.add_node('A', **VariableDefinition(
            name='A',
            description='A',
            type='int',
            values=[0, 1, 2, 3]
        ).to_dict())
        G.add_node('B', **VariableDefinition(
            name='B',
            description='B',
            type='int',
            values=[0, 1, 2, 3],
            current_value=0,
            contextual_information='context',
            supporting_text_snippets=['text1', 'text2']
        ).to_dict())
        G.add_node('C', **InferredVariableDefinition(
            name='C',
            description='C',
            type='int',
            values=[0, 1, 2, 3],
            current_value=0,
            contextual_information='context',
            supporting_text_snippets=['text1', 'text2'],
            causal_effect='causal_effect'
        ).to_dict())
        return G
    
    def test_update_graph_node_values_in_place(self, graph):
        old_graph = graph.copy()
        new_value = VariableDefinition(
            name='A',
            description='A2',
            type='float',
            values=[4, 5, 6, 7],
            current_value=2,
            contextual_information='context2',
            supporting_text_snippets=['text3', 'text4']
        ).to_dict()
        
        update_graph_node_values_in_place(graph, new_value)
        
        assert graph.nodes['B'] == old_graph.nodes['B']
        assert graph.nodes['C'] == old_graph.nodes['C']
        assert graph.nodes['A'] is not old_graph.nodes['A']
        assert graph.nodes['A'] is not new_value
        assert graph.nodes['A']['name'] == 'A'
        assert graph.nodes['A']['description'] == 'A2'
        assert graph.nodes['A']['type'] == 'float'
        assert graph.nodes['A']['values'] == [4, 5, 6, 7]
        assert graph.nodes['A']['current_value'] == 2
        assert graph.nodes['A']['contextual_information'] == 'context2'
        assert graph.nodes['A']['supporting_text_snippets'] == ['text3', 'text4']
        assert graph.nodes['A']['causal_effect'] == 2

    def test_update_graph_node_values_in_place_2(self, graph):
        old_graph = graph.copy()
        new_value = VariableDefinition(
            name='B',
            description='B2',
            type='float',
            values=[4, 5, 6, 7],
            current_value=2,
            contextual_information='context2',
            supporting_text_snippets=['text3', 'text4']
        ).to_dict()
        
        update_graph_node_values_in_place(graph, new_value)
        
        assert graph.nodes['A'] == old_graph.nodes['A']
        assert graph.nodes['C'] == old_graph.nodes['C']
        assert graph.nodes['B'] is not old_graph.nodes['B']
        assert graph.nodes['B'] is not new_value
        assert graph.nodes['B']['name'] == 'B'
        assert graph.nodes['B']['description'] == 'B2'
        assert graph.nodes['B']['type'] == 'float'
        assert graph.nodes['B']['values'] == [4, 5, 6, 7]
        assert graph.nodes['B']['current_value'] == 2
        assert graph.nodes['B']['contextual_information'] == 'context2'
        assert graph.nodes['B']['supporting_text_snippets'] == ['text3', 'text4']
        assert graph.nodes['B']['causal_effect'] == 2

    def test_update_graph_node_values_in_place_with_inferred_variable(self, graph):
        old_graph = graph.copy()
        new_value = InferredVariableDefinition(
            name='C',
            description='C2',
            type='float',
            values=[4, 5, 6, 7],
            current_value=2,
            contextual_information='context2',
            supporting_text_snippets=['text3', 'text4'],
            causal_effect='causal_effect2'
        ).to_dict()
        
        update_graph_node_values_in_place(graph, new_value)
        
        assert graph.nodes['A'] == old_graph.nodes['A']
        assert graph.nodes['B'] == old_graph.nodes['B']
        assert graph.nodes['C'] is not old_graph.nodes['C']
        assert graph.nodes['C'] is not new_value
        assert graph.nodes['C']['name'] == 'C'
        assert graph.nodes['C']['description'] == 'C2'
        assert graph.nodes['C']['type'] == 'float'
        assert graph.nodes['C']['values'] == [4, 5, 6, 7]
        assert graph.nodes['C']['current_value'] == 2
        assert graph.nodes['C']['contextual_information'] == 'context2'
        assert graph.nodes['C']['supporting_text_snippets'] == ['text3', 'text4']
        assert graph.nodes['C']['causal_effect'] == 'causal_effect2'

    def test_update_graph_node_values_in_place_with_nonexistent_node(self, graph):
        pytest.raises(KeyError, 
            update_graph_node_values_in_place,
            graph,
            VariableDefinition(
                name='D',
                description='D',
                type='int',
                values=[0, 1, 2, 3],
                current_value=2,
                contextual_information='context2',
                supporting_text_snippets=['text3', 'text4']
            ).to_dict()
        )

    def test_update_graph_node_values_in_place_no_causal_effect(self, graph):
        old_graph = graph.copy()
        new_value = VariableDefinition(
            name='A',
            description='A2',
            type='float',
            values=[4, 5, 6, 7],
            current_value=2,
            contextual_information='context2',
            supporting_text_snippets=['text3', 'text4']
        ).to_dict()
        
        update_graph_node_values_in_place(graph, new_value, add_causal_effect=False)
        
        assert graph.nodes['B'] == old_graph.nodes['B']
        assert graph.nodes['C'] == old_graph.nodes['C']
        assert graph.nodes['A'] is not old_graph.nodes['A']
        assert graph.nodes['A'] is not new_value
        assert graph.nodes['A']['name'] == 'A'
        assert graph.nodes['A']['description'] == 'A2'
        assert graph.nodes['A']['type'] == 'float'
        assert graph.nodes['A']['values'] == [4, 5, 6, 7]
        assert graph.nodes['A']['current_value'] == 2
        assert graph.nodes['A']['contextual_information'] == 'context2'
        assert graph.nodes['A']['supporting_text_snippets'] == ['text3', 'text4']
        assert 'causal_effect' not in graph.nodes['B']

    def test_update_node_values_in_place_no_causal_effect_with_inferred_variable(self, graph):
        old_graph = graph.copy()
        new_value = InferredVariableDefinition(
            name='C',
            description='C2',
            type='float',
            values=[4, 5, 6, 7],
            current_value=2,
            contextual_information='context2',
            supporting_text_snippets=['text3', 'text4'],
            causal_effect='causal_effect2'
        ).to_dict()
        
        update_graph_node_values_in_place(graph, new_value, add_causal_effect=False)
        
        assert graph.nodes['A'] == old_graph.nodes['A']
        assert graph.nodes['B'] == old_graph.nodes['B']
        assert graph.nodes['C'] is not old_graph.nodes['C']
        assert graph.nodes['C'] is not new_value
        assert graph.nodes['C']['name'] == 'C'
        assert graph.nodes['C']['description'] == 'C2'
        assert graph.nodes['C']['type'] == 'float'
        assert graph.nodes['C']['values'] == [4, 5, 6, 7]
        assert graph.nodes['C']['current_value'] == 2
        assert graph.nodes['C']['contextual_information'] == 'context2'
        assert graph.nodes['C']['supporting_text_snippets'] == ['text3', 'text4']
        assert graph.nodes['C']['causal_effect'] == 'causal_effect2'

    def test_update_graph_node_values_in_place_reset_old_values(self, graph):
        old_graph = graph.copy()
        new_value = VariableDefinition(
            name='A',
            description='A2',
            type='float',
            values=[4, 5, 6, 7],
            current_value=2,
        ).to_dict()
        
        update_graph_node_values_in_place(graph, new_value, reset_old_values=True)
        
        assert graph.nodes['B'] == old_graph.nodes['B']
        assert graph.nodes['C'] == old_graph.nodes['C']
        assert graph.nodes['A'] is not old_graph.nodes['A']
        assert graph.nodes['A'] is not new_value
        assert graph.nodes['A']['name'] == 'A'
        assert graph.nodes['A']['description'] == 'A2'
        assert graph.nodes['A']['type'] == 'float'
        assert graph.nodes['A']['values'] == [4, 5, 6, 7]
        assert graph.nodes['A']['current_value'] == 2
        assert 'contextual_information' not in graph.nodes['A']
        assert 'supporting_text_snippets' not in graph.nodes['A']
        assert graph.nodes['A']['causal_effect'] == 2

    def test_update_graph_node_values_in_place_reset_old_values_with_inferred_variable(self, graph):
        old_graph = graph.copy()
        new_value = InferredVariableDefinition(
            name='C',
            description='C2',
            type='float',
            values=[4, 5, 6, 7],
            current_value=2,
            causal_effect='causal_effect2'
        ).to_dict()
        
        update_graph_node_values_in_place(graph, new_value, reset_old_values=True)
        
        assert graph.nodes['A'] == old_graph.nodes['A']
        assert graph.nodes['B'] == old_graph.nodes['B']
        assert graph.nodes['C'] is not old_graph.nodes['C']
        assert graph.nodes['C'] is not new_value
        assert graph.nodes['C']['name'] == 'C'
        assert graph.nodes['C']['description'] == 'C2'
        assert graph.nodes['C']['type'] == 'float'
        assert graph.nodes['C']['values'] == [4, 5, 6, 7]
        assert graph.nodes['C']['current_value'] == 2
        assert 'contextual_information' not in graph.nodes['C']
        assert 'supporting_text_snippets' not in graph.nodes['C']
        assert graph.nodes['C']['causal_effect'] == 'causal_effect2'

        
class TestBuildIntervenedGraph:

    @pytest.fixture
    def graph_structure(self):
        G = nx.DiGraph()
        G.add_edges_from([
            ('A', 'B'),
            ('A', 'C'),
            ('B', 'D'),
            ('C', 'D')
        ])
        return G
    
    @pytest.fixture
    def graph_values(self):
        G = nx.DiGraph()
        G.add_node('A', **VariableDefinition(
            name='A',
            description='A',
            type='int',
            values=[0, 1, 2, 3]
        ).to_dict())
        G.add_node('B', **VariableDefinition(
            name='B',
            description='B',
            type='int',
            values=[0, 1, 2, 3],
            current_value=0,
            contextual_information='context',
            supporting_text_snippets=['text1', 'text2']
        ).to_dict())
        G.add_node('C', **VariableDefinition(
            name='C',
            description='C',
            type='int',
            values=[0, 1, 2, 3],
            current_value=0,
            contextual_information='context',
            supporting_text_snippets=['text1', 'text2']
        ).to_dict())
        G.add_node('D', **VariableDefinition(
            name='D',
            description='D',
            type='int',
            values=[0, 1, 2, 3],
            current_value=0,
            contextual_information='context',
            supporting_text_snippets=['text1', 'text2']
        ).to_dict())
        G.add_edges_from([
            ('A', 'B'),
            ('A', 'C'),
            ('B', 'D'),
            ('C', 'D')
        ])
        return G
    
    @pytest.fixture
    def interventions(self):
        return [
            {'name': 'B', 'description': 'B2', 'type': 'int', 'values': [4, 5, 6, 7], 'current_value': 2},
            {'name': 'C', 'description': 'C2', 'type': 'int', 'values': [4, 5, 6, 7], 'current_value': 2}
        ]
    
    def test_build_intervened_graph_structure_only(self, graph_structure, interventions):
        intervened_graph = build_intervened_graph(graph_structure, interventions, structure_only=True)
        
        assert 'A' in intervened_graph.nodes
        assert 'B' in intervened_graph.nodes
        assert 'C' in intervened_graph.nodes
        assert 'D' in intervened_graph.nodes

        assert intervened_graph.nodes['B'] == graph_structure.nodes['B']
        assert intervened_graph.nodes['C'] == graph_structure.nodes['C']
        assert intervened_graph.nodes['D'] == graph_structure.nodes['D']

        assert list(intervened_graph.edges) == [
            ('B', 'D'),
            ('C', 'D')
        ]

    def test_build_intervened_graph(self, graph_values, interventions):
        intervened_graph = build_intervened_graph(graph_values, interventions)
        
        assert 'A' in intervened_graph.nodes
        assert 'B' in intervened_graph.nodes
        assert 'C' in intervened_graph.nodes
        assert 'D' in intervened_graph.nodes

        assert list(intervened_graph.edges) == [
            ('B', 'D'),
            ('C', 'D')
        ]

        assert intervened_graph.nodes['B']['name'] == 'B'
        assert intervened_graph.nodes['B']['description'] == 'B2'
        assert intervened_graph.nodes['B']['type'] == 'int'
        assert intervened_graph.nodes['B']['values'] == [4, 5, 6, 7]
        assert intervened_graph.nodes['B']['current_value'] == 2

        assert intervened_graph.nodes['C']['name'] == 'C'
        assert intervened_graph.nodes['C']['description'] == 'C2'
        assert intervened_graph.nodes['C']['type'] == 'int'
        assert intervened_graph.nodes['C']['values'] == [4, 5, 6, 7]
        assert intervened_graph.nodes['C']['current_value'] == 2