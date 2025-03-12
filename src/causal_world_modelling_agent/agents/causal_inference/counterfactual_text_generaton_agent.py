
from typing import Optional, Dict

from smolagents import CodeAgent

from ..factory import AgentFactory




class CounterfactualTextGenerationAgent(CodeAgent):

    def run(self, *args, additional_args: Optional[Dict] = None, **kwargs):
        if "causal_graph" not in additional_args:
            raise ValueError("Causal graph must be provided as an argument")
        causal_graph = additional_args["causal_graph"]

        modified_variables = {}
        for node, attrs in causal_graph.nodes(data=True):
            if "causal_effect" in attrs:
                modified_variables[node] = attrs["causal_effect"] 
        
        return super().run(*args, additional_args=modified_variables, **kwargs)
            



class CounterfactualTextGenerationAgentFactory(AgentFactory[CounterfactualTextGenerationAgent]):

    def __init__(self, path_to_system_prompt: str = 'counterfactual_text_generation.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(CounterfactualTextGenerationAgent, path_to_system_prompt, use_prompt_lib_folder)