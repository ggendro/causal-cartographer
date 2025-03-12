
import networkx as nx
import pytest

from causal_world_modelling_agent.utils.graph_utils import is_digraph
from mocks.mock_models import UpdateGraphMockModel
from causal_world_modelling_agent.agents.causal_discovery.self_iterative_agent import SelfIterativeDiscoveryAgentFactory




class TestIterativeAgent:

    @pytest.fixture
    def iterative_agent(self):
        return SelfIterativeDiscoveryAgentFactory().createAgent(UpdateGraphMockModel())
    
    @pytest.fixture
    def graph(self):
        graph = nx.DiGraph()
        graph.add_node("A", description="Node A")
        graph.add_node("B", description="Node B")
        graph.add_edge("A", "B", description="Edge A -> B")
        return graph
    
    @pytest.fixture
    def iterative_agent_with_data(self, graph):
        return SelfIterativeDiscoveryAgentFactory(initial_graph=graph).createAgent(UpdateGraphMockModel())
    
    @pytest.fixture
    def iterative_agent_with_history(self, graph):
        return SelfIterativeDiscoveryAgentFactory(previous_history=[graph]).createAgent(UpdateGraphMockModel())
    
    def test_run_checks(self, iterative_agent):
        assert is_digraph(iterative_agent.run("Hello world!"))
    
    def test_run(self, iterative_agent):
        updated_graph = iterative_agent.run("Hello world!")

        assert set(updated_graph.nodes) == {"node1", "node2"}
        assert set(updated_graph.edges) == {("node1", "node2")}

    def test_run_with_data(self, iterative_agent_with_data):
        updated_graph = iterative_agent_with_data.run("Hello world!")

        assert set(updated_graph.nodes) == {"A", "B", "node1", "node2"}
        assert set(updated_graph.edges) == {("A", "B"), ("node1", "node2")}