
from ..factory import AgentFactory
from ...syntax.messages import EVENT, OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP
from ..custom_prompt_agent import CustomPromptAgent


class AtomicDiscoveryAgentFactory(AgentFactory[CustomPromptAgent]):

    def __init__(self, path_to_system_prompt: str = 'atomic_causal_discovery.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(CustomPromptAgent, path_to_system_prompt, use_prompt_lib_folder)

    def createAgent(self, base_model, *args, **kwargs) -> CustomPromptAgent:
        return super().createAgent(
            base_model,
            *args,
            additional_system_prompt_variables={
                'event': EVENT,
                'observed_variable': OBSERVED_VARIABLE,
                'variable': VARIABLE,
                'causal_relationship': CAUSAL_RELATIONSHIP
            },
            **kwargs
        )