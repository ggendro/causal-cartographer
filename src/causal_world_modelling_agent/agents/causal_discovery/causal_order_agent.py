
from smolagents import Model

from ...core.agent import CustomSystemPromptCodeAgent
from ..factory import AgentFactory
from ...tools.causal_order import is_a_valid_partial_order
from ...syntax.messages import VARIABLE




class CausalOrderAgentFactory(AgentFactory):

    AGENT_NAME = "causal_order_agent"

    DESCRIPTION = """Agent that builds a partial order fom a list of causal variables and verifies that the order is valid. Variables are provided in the additional arguments as a list with name 'causal_variables'. The agent returns a list of tuples, where each tuple corresponds to a pair of non-identical variable names that are ordered. 
                    For example, if the list is [(a, b), (b, c)], it means that a < b < c.
                    Example of agent call:
                    ```
                    order_list = {agent_name}(task="...", additional_args={{'causal_variables':[{{...}}, {{...}}, ...]}})
                    ```
                    """

    INTERNAL_SYSTEM_PROMPT = f"""You are an agent that builds a partial order fom a list of causal variables and verifies that the order is valid. The returned order should not create cycles and only use valid elements of the original list. 
                    The agent must use common sense knowledge to estimate the order of the causal variables. Variables are provided as an argument list with name 'causal_variables'. Causal variables have the following format:
                    {VARIABLE}
                    The agent must return a list of tuples, where each tuple corresponds to a pair of non-identical variable names that should be ordered. For example, if the list is [(a, b), (b, c)], it means that a < b < c.
                    Here an example:
                    ---
                    Task: 
                    """ # TODO: complete the example, analyse the answer from the notebook and update pipeline

    def createAgent(self, base_model: Model) -> CustomSystemPromptCodeAgent: # TODO: switch to final_answer_checks argument
        return CustomSystemPromptCodeAgent(
                    tools=[is_a_valid_partial_order], 
                    model=base_model, 
                    name="causal_order_agent", 
                    description=CausalOrderAgentFactory.DESCRIPTION.format(agent_name=CausalOrderAgentFactory.AGENT_NAME),
                    custom_system_prompt=CausalOrderAgentFactory.INTERNAL_SYSTEM_PROMPT
            )