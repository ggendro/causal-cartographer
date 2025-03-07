
from ..factory import AgentFactory
from ...syntax.messages import EVENT, OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP


class AtomicDiscoveryAgentFactory(AgentFactory):

    def __init__(self, path_to_prompt_syntax: str = 'atomic_causal_discovery.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(path_to_prompt_syntax, use_prompt_lib_folder)

        self.additional_system_prompt = self.additional_system_prompt.format(
            event=EVENT,
            observed_variable=OBSERVED_VARIABLE,
            variable=VARIABLE,
            causal_relationship=CAUSAL_RELATIONSHIP
        )