
import pytest

from mocks.mock_models import DummyMockModel
from causal_world_modelling_agent.agents.causal_discovery.atomic_discovery_agent import AtomicDiscoveryAgentFactory




class TestDiscoveryAgent:

    @pytest.fixture
    def discovery_agent(self):
        return AtomicDiscoveryAgentFactory().createAgent(DummyMockModel())
    
    def test_prompt_templates(self, discovery_agent):
        assert discovery_agent.prompt_templates['system_prompt'].startswith("You are an agent that extracts a networkx causal graph from a text snippet.")
    
    def test_run(self, discovery_agent):
        assert discovery_agent.run("Hello world!") == "Dummy answer!"