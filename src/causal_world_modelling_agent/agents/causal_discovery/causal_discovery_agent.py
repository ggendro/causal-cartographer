
from typing import List
import json
import networkx as nx

from smolagents import Model

from ...core.agent import CustomSystemPromptCodeAgent
from ...core.definitions import Message
from ...syntax.messages import OBSERVED_VARIABLE
from ..factory import AgentFactory




class PairwiseCausalDiscoveryAgent(CustomSystemPromptCodeAgent):

    def run(self, *args, causal_variables: List[Message], **kwargs) -> nx.DiGraph:
        graph = nx.DiGraph()

        for causal_variable in causal_variables:
            graph.add_node(causal_variable['name'], **causal_variable)

        for i, source_variable in enumerate(causal_variables):
            for j, target_variable in enumerate(causal_variables[i+1:]):
                    is_causal_relationship = super().run(*args, additional_args={'source_variable': ..., 'target_variable': ...} **kwargs)

                    if is_causal_relationship in ['True', 'true']:
                        graph.add_edge(source_variable['name'], target_variable['name']) # TODO: switch to tool_parser and grammer arguments

        return graph
        


class CausalDiscoveryAgentFactory(AgentFactory):

    AGENT_NAME = "causal_discovery_agent"

    DESCRIPTION = """Agent that determines the causal relationships that exist in a set of causal variables. Variables are provided in the additional arguments as a list with name 'causal_variables'. The agent returns a networkx DiGaph object.
                    Example of agent call:
                    ```
                    causal_graph = {agent_name}(task="...", additional_args={{'causal_variables':[{{...}}, {{...}}, ...]}})
                    ```
                    """

    SYSTEM_PROMPT = f"""You are an agent that determines if a causal relationship exists between two causal variables. Variables are provided as arguments with names 'source_variable' and 'target_variable'.
                    The agent must use common sense knowledge to estimate the causal relationship between the variables. Causal variables have the following format:
                    {OBSERVED_VARIABLE}
                    The agent must return a single boolean value indicating if a causal relationship exists between the two variables as a final answer.
                    Example:
                    ```
                    final_answer(True)
                    ```
                    or
                    ```
                    final_answer(False)
                    ```            
                    """

    def createAgent(self, base_model: Model) -> PairwiseCausalDiscoveryAgent:
        return PairwiseCausalDiscoveryAgent(
                    tools=[], 
                    model=base_model, 
                    additional_authorized_imports=[],
                    name=CausalDiscoveryAgentFactory.AGENT_NAME, 
                    description=CausalDiscoveryAgentFactory.DESCRIPTION.format(agent_name=CausalDiscoveryAgentFactory.AGENT_NAME),
                    custom_system_prompt=CausalDiscoveryAgentFactory.SYSTEM_PROMPT
                )