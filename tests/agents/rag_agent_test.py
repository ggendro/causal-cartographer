
import pytest

from causal_world_modelling_agent.utils.graph_utils import is_digraph
from mocks.mock_models import DummyMockModel, CreateGraphMockModel
from causal_world_modelling_agent.agents.causal_discovery.atomic_rag_agent import AtomicRAGDiscoveryAgentFactory




class TestRAGAgent:

    @pytest.fixture
    def rag_agent(self):
        agent = AtomicRAGDiscoveryAgentFactory().createAgent(DummyMockModel(), api_key="dummy")
        tool = lambda x: "Dummy answer!"
        tool.name = "graph_retriever"
        tool.description = "A tool that retrieves graphs."
        tool.inputs = 'string'
        tool.output_type = 'string'
        agent.tools["graph_retriever"] = tool
        agent.final_answer_checks = []

        return agent
    
    @pytest.fixture
    def rag_agent_with_check(self):
        agent = AtomicRAGDiscoveryAgentFactory().createAgent(CreateGraphMockModel(), api_key="dummy")
        tool = lambda x: "Dummy answer!"
        tool.name = "graph_retriever"
        tool.description = "A tool that retrieves graphs."
        tool.inputs = 'string'
        tool.output_type = 'string'
        agent.tools["graph_retriever"] = tool

        return agent
    
    def test_graph_tool_exists(self, rag_agent):
        assert "graph_retriever" in rag_agent.tools
    
    def test_run(self, rag_agent):
        assert rag_agent.run("Hello world!") == "Dummy answer!"

    def test_run_with_check(self, rag_agent_with_check):
        assert is_digraph(rag_agent_with_check.run("Hello world!"))