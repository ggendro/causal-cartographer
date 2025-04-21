
import networkx as nx
import pytest
import re

from causal_world_modelling_agent.world_model.world_manager import extract_world_nodes, ground_in_world


class TestGrounding:

    @pytest.fixture
    def worlds_graph(self):
        G = nx.DiGraph()
        G.add_node("A", base_attribute_1="value_1", base_attribute_2="value_2", world_0={"attr_world_0": 0}, world_1="attr_world_1")
        G.add_node("B", base_attribute_1="value_3", base_attribute_2="value_4", world_0={"attr_world_0": 0}, world_1="attr_world_1")
        G.add_node("C", base_attribute_2="value_5", base_attribute_3="value_6", world_1="attr_world_0", world_2="attr_world_1")
        G.add_node("D", base_attribute_1="value_7", base_attribute_3="value_8")
        G.add_node("E", world_17="attr_world_0")
        G.add_node("F", world_20="attr_world_1", world_21="attr_world_1", world_22="attr_world_1", world_0={"attr_world_0": 0})
        G.add_node("G", xworld_0="attr_world_0")
        G.add_node("H", world_0x="attr_world_0")
        G.add_edges_from([("A", "B"), ("B", "C"), ("C", "D"), ("D", "F"), ("F", "G"), ("G", "H")])
        return G
    

    def test_extract_world_nodes_empty_graph(self):
        G = nx.DiGraph()
        world_nodes = extract_world_nodes(G, r"world_\d+")
        assert len(world_nodes) == 0
    
    def test_extract_world_nodes(self, worlds_graph):
        expected = {
            "world_0": set(["A", "B", "F"]),
            "world_1": set(["A", "B", "C"]),
            "world_2": set(["C"]),
            "world_17": set(["E"]),
            "world_20": set(["F"]),
            "world_21": set(["F"]),
            "world_22": set(["F"])
        }
        world_nodes = extract_world_nodes(worlds_graph, r"world_\d+")

        assert len(world_nodes) == 7
        assert world_nodes == expected

    def test_extract_world_nodes_no_matching_nodes(self, worlds_graph):
        world_nodes = extract_world_nodes(worlds_graph, r"warld_\d+")
        assert len(world_nodes) == 0

    def test_extract_world_nodes_partial_match(self, worlds_graph):
        expected = {
            "world_0": set(["A", "B", "F"]),
            "world_1": set(["A", "B", "C"]),
            "world_2": set(["C"])
        }
        world_nodes = extract_world_nodes(worlds_graph, r"world_\d")
        assert len(world_nodes) == 3
        assert world_nodes == expected

    def test_ground_in_world_empty_graph(self):
        G = nx.DiGraph()
        result = ground_in_world(G, "world_0", r"world_\d+")
        assert len(result.nodes) == 0
        assert len(result.edges) == 0

    def test_ground_in_world(self, worlds_graph):
        expected_nodes = {'A', 'B', 'F'}
        key = r"world_\d+"
        result = ground_in_world(worlds_graph, "world_0", key)
        assert len(result.nodes) == len(worlds_graph.nodes)
        assert len(result.edges) == len(worlds_graph.edges)
        assert all(node in result.nodes for node in worlds_graph.nodes)
        assert all(edge in result.edges for edge in worlds_graph.edges)

        for node, attrs in result.nodes(data=True):
            for attr in attrs:
                assert re.fullmatch(key, attr) is None
                if attr == "attr_world_0":
                    assert result.nodes[node][attr] == worlds_graph.nodes[node]['world_0'][attr] # grounded attribute of world 0
                else:
                    assert result.nodes[node][attr] == worlds_graph.nodes[node][attr]

    def test_ground_in_world_no_matching_nodes(self, worlds_graph):
        key = r"world_\d+"
        result = ground_in_world(worlds_graph, "world_100", key)
        assert len(result.nodes) == len(worlds_graph.nodes)
        assert len(result.edges) == len(worlds_graph.edges)
        assert all(node in result.nodes for node in worlds_graph.nodes)
        assert all(edge in result.edges for edge in worlds_graph.edges)

        for node, attrs in result.nodes(data=True):
            for attr in attrs:
                assert re.fullmatch(key, attr) is None
                assert result.nodes[node][attr] == worlds_graph.nodes[node][attr]

    def test_ground_in_world_no_matching_key(self, worlds_graph):
        key = r"warld_\d+"
        result = ground_in_world(worlds_graph, "world_100", key)
        assert len(result.nodes) == len(worlds_graph.nodes)
        assert len(result.edges) == len(worlds_graph.edges)
        assert all(node in result.nodes for node in worlds_graph.nodes)
        assert all(edge in result.edges for edge in worlds_graph.edges)

        for node, attrs in result.nodes(data=True):
            assert len(attrs) == len(worlds_graph.nodes[node])
            for attr in attrs:
                assert result.nodes[node][attr] == worlds_graph.nodes[node][attr]

    def test_ground_in_world_none(self, worlds_graph):
        key = r"world_\d+"
        result = ground_in_world(worlds_graph, None, key)
        assert len(result.nodes) == len(worlds_graph.nodes)
        assert len(result.edges) == len(worlds_graph.edges)
        assert all(node in result.nodes for node in worlds_graph.nodes)
        assert all(edge in result.edges for edge in worlds_graph.edges)

        for node, attrs in result.nodes(data=True):
            for attr in attrs:
                assert re.fullmatch(key, attr) is None
                assert result.nodes[node][attr] == worlds_graph.nodes[node][attr]
