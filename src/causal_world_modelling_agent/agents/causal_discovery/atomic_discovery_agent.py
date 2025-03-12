
from ...utils.graph_utils import isDigraph
from ...utils.message_utils import isGraphMessageDefinition
from ..factory import AgentFactory
from ...syntax.definitions import EventDefinition, ObservedVariableDefinition, VariableDefinition, CausalRelationshipDefinition
from ..custom_prompt_agent import CustomPromptAgent




class AtomicDiscoveryAgentFactory(AgentFactory[CustomPromptAgent]):

    def __init__(self, path_to_system_prompt: str = 'atomic_causal_discovery.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(CustomPromptAgent, path_to_system_prompt, use_prompt_lib_folder)

    def createAgent(self, base_model, *args, **kwargs) -> CustomPromptAgent:
        return super().createAgent(
            base_model,
            *args,
            additional_system_prompt_variables={
                'event': EventDefinition.get_definition(),
                'observed_variable': ObservedVariableDefinition.get_definition(),
                'variable': VariableDefinition.get_definition(),
                'causal_relationship': CausalRelationshipDefinition.get_definition()
            },
            final_answer_checks=[isDigraph, isGraphMessageDefinition],
            **kwargs
        )