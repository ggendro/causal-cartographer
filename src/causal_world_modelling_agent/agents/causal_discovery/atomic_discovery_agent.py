
from typing import Optional

from smolagents import Model

from ...core.agent import CustomSystemPromptCodeAgent
from ..factory import AgentFactory
from ...syntax.messages import EVENT, OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP


class AtomicDiscoveryAgentFactory(AgentFactory):

    def __init__(self, path_to_prompt_syntax: Optional[str] = None):
        if not path_to_prompt_syntax:
            import os
            path_to_prompt_syntax = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'syntax', 'prompts', 'atomic_causal_discovery.yaml')

        super().__init__(path_to_prompt_syntax)

        self.system_prompt = self.system_prompt.format(
            event=EVENT,
            observed_variable=OBSERVED_VARIABLE,
            variable=VARIABLE,
            causal_relationship=CAUSAL_RELATIONSHIP
        )

    def createAgent(self, base_model: Model) -> CustomSystemPromptCodeAgent:
        return CustomSystemPromptCodeAgent(
            tools=[], 
            model=base_model, 
            additional_authorized_imports=["networkx"],
            name="end_to_end_causal_extraction_agent", 
            description=self.description,
            custom_system_prompt=self.system_prompt,
        )