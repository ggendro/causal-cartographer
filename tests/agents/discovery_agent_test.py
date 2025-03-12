
import pytest

from causal_world_modelling_agent.utils.graph_utils import isDigraph
from mocks.mock_models import DummyMockModel, CreateGraphMockModel
from causal_world_modelling_agent.agents.causal_discovery.atomic_discovery_agent import AtomicDiscoveryAgentFactory




class TestDiscoveryAgent:

    @pytest.fixture
    def discovery_agent(self):
        agent = AtomicDiscoveryAgentFactory().createAgent(DummyMockModel())
        agent.final_answer_checks = []
        return agent
    
    @pytest.fixture
    def discovery_agent_with_check(self):
        agent = AtomicDiscoveryAgentFactory().createAgent(CreateGraphMockModel())
        agent.final_answer_checks = [isDigraph]
        return agent
    
    def test_prompt_templates(self, discovery_agent):
        assert discovery_agent.prompt_templates['system_prompt'].startswith("You are an agent that extracts a networkx causal graph from a text snippet.")
    
    def test_run(self, discovery_agent):
        assert discovery_agent.run("Hello world!") == "Dummy answer!"

    def test_run_with_check(self, discovery_agent_with_check):
        assert isDigraph(discovery_agent_with_check.run("Hello world!"))