
from typing import Dict, Optional, Any, List
import networkx as nx

from ..factory import AgentFactory
from ..custom_prompt_agent import CustomPromptAgent
from ...syntax.definitions import Message, InferredVariableDefinition, CausalRelationshipDefinition
from ...utils.message_utils import isInferredVariableDefinition, isCausalRelationshipDefinition


class CoTAgent(CustomPromptAgent):
    """
    CoT agent for causal inference tasks.
    """

    @classmethod
    def _build_user_prompt(cls, task, causal_graph, target_variable, observations, interventions, is_counterfactual) -> str:
                
        task = f'{task}\n\n'

        for node, attrs in causal_graph.nodes(data=True):
            task += f"The variable '{node}' (type: {attrs['type']}, with possible values: {attrs['values']}) can be described as follows:\n{attrs['description']}\n"
        task += "\n"

        for source, target, attrs in causal_graph.edges(data=True):
            task += f"The variable '{source}' causes the variable '{target}' with the following relationship (type: {attrs['type']}):\n{attrs['description']}\n"

        if len(observations) > 0:
            task += "\nWe observe the following values for the variables:\n"
            for observation in observations:
                task += f"The observed value of variable '{observation['name']}' is {observation['current_value']}.\n"

        if len(interventions) > 0:
            task += "\nWe intervene on the following variables and set their values:\n"
            for intervention in interventions:
                task += f"We set the value of variable '{intervention['name']}' to {intervention['current_value']}.\n"

        task += f"\nWe want to esimate the value of the target variable: '{target_variable}'.\n\n"
        task += "Please provide the estimated value of the target variable, along with the causal graph that supports your answer. Follow the provided instructions and respect the requested answer format."

        return task

    def run(self, task, *args, additional_args: Optional[Dict] = None, **kwargs):
        if "causal_graph" not in additional_args:
            raise ValueError("Causal graph must be provided as an argument")
        causal_graph = additional_args.pop("causal_graph")
        
        if "target_variable" not in additional_args:
            raise ValueError("Target variable must be provided as an argument")
        target_variable = additional_args.pop("target_variable")
        
        if "observations" not in additional_args or additional_args["observations"] is None:
            observations = []
            if "observations" in additional_args:
                del additional_args["observations"]
        else:
            observations = additional_args.pop("observations")
        
        if "interventions" not in additional_args or additional_args["interventions"] is None:
            interventions = []
            if "interventions" in additional_args:
                del additional_args["interventions"]
        else:
            interventions = additional_args.pop("interventions")

        if target_variable not in causal_graph.nodes:
            raise ValueError("Target variable not found in causal graph")

        if "is_counterfactual" not in additional_args:
            is_counterfactual = False
        else:
            is_counterfactual = additional_args.pop("is_counterfactual")
        
        if not is_counterfactual and target_variable in [observation["name"] for observation in observations]:
            raise ValueError("Target variable cannot be observed unless in counterfactual mode")
        
        if target_variable in [intervention["name"] for intervention in interventions]:
            raise ValueError("Target variable cannot be intervened on")

        for node in causal_graph.nodes:
            if "current_value" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["current_value"]
            if "contextual_information" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["contextual_information"]
            if "causal_effect" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["causal_effect"]

        task = self._build_user_prompt(task, causal_graph, target_variable, observations, interventions, is_counterfactual)

        return super().run(task, *args, additional_args=additional_args, **kwargs)  # don't forget to ask to return both the answer nd the causal graph (with the right format) in the system prompt



def isAnswerValid(message: Any, memory: Optional[List[Any]] = None) -> bool:
    if not isinstance(message, tuple):
        raise ValueError("Final answer must be a tuple containing the answer as a string and the causal graph as a networkx DiGraph")
    if not len(message) == 2:
        raise ValueError(f"Final answer must be a tuple of length 2, but got {len(message)}")
    if not isinstance(message[0], str):
        raise ValueError(f"The first element of the final answer tuple must be a string, but got {type(message[0])}")
    if not isinstance(message[1], nx.DiGraph):
        raise ValueError(f"The second element of the final answer tuple must be a networkx DiGraph, but got {type(message[1])}")
    
    for node, data in message[1].nodes(data=True):
        try:
            isInferredVariableDefinition(data)
        except Exception as e:
            raise ValueError(f"Node '{node}' in the causal graph does not match the expected variable definition: {e}")
    
    for cause, effect, data in message[1].edges(data=True):
        try:
            isCausalRelationshipDefinition(data)
        except Exception as e:
            raise ValueError(f"Edge from '{cause}' to '{effect}' in the causal graph does not match the expected causal relationship definition: {e}")

    return True
        

class CausalCoTAgentFactory(AgentFactory[CoTAgent]):
    """
    Factory class for creating Causal CoT agents.
    """

    def __init__(self, path_to_system_prompt: str = 'causal_cot.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(CoTAgent, path_to_system_prompt, use_prompt_lib_folder)

    def createAgent(self, *args, **kwargs) -> CoTAgent:
        return super().createAgent(
            *args,
            additional_system_prompt_variables={
                'variable': InferredVariableDefinition.get_definition(),
                'causal_relationship': CausalRelationshipDefinition.get_definition()
            },
            final_answer_checks=[isAnswerValid],
            **kwargs
        )