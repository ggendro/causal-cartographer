
from typing import Optional, Dict, Tuple, List

import networkx as nx
from smolagents import Model

from ..factory import AgentFactory
from ...core.agent import CustomSystemPromptCodeAgent




class CounterfactualTextGenerationAgent(CustomSystemPromptCodeAgent):

    def run(self, *args, additional_args: Optional[Dict] = None, **kwargs):
        if "causal_graph" not in additional_args:
            raise ValueError("Causal graph must be provided as an argument")
        causal_graph = additional_args["causal_graph"]

        modified_variables = {}
        for node, attrs in causal_graph.nodes(data=True):
            if "causal_effect" in attrs:
                modified_variables[node] = attrs["causal_effect"] 
        
        return super().run(*args, additional_args=modified_variables, **kwargs)
            



class CounterfactualTextGenerationAgentFactory(AgentFactory):

    def __init__(self, path_to_prompt_syntax: str = 'counterfactual_text_generation.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(path_to_prompt_syntax, use_prompt_lib_folder)

    def createAgent(self, base_model: Model) -> CounterfactualTextGenerationAgent:
        return CounterfactualTextGenerationAgent(
                    tools=[],
                    model=base_model, 
                    additional_authorized_imports=[],
                    name=self.name, 
                    description=self.description,
                    custom_system_prompt=self.additional_system_prompt,
                    managed_agents=[]
        )