
from typing import Optional
import yaml

from smolagents import Model

from ..core.agent import CustomSystemPromptCodeAgent

class AgentFactory:

    def __init__(self, path_to_prompt_syntax: Optional[str] = None, use_prompt_lib_folder: bool = True):
        self.additional_system_prompt = ''
        self.description = ''
        self.user_pre_prompt = ''
        self.name = ''

        if path_to_prompt_syntax:
            if use_prompt_lib_folder:
                import os
                path_to_prompt_syntax = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'syntax', 'prompts', path_to_prompt_syntax)

            self._readPromptSyntax(path_to_prompt_syntax)

    def _readPromptSyntax(self, path_to_prompt_syntax: str):
        with open(path_to_prompt_syntax, 'r') as file:
            prompt_syntax = yaml.safe_load(file)
            if 'additional_system_prompt' in prompt_syntax:
                self.additional_system_prompt = prompt_syntax['additional_system_prompt']
            if 'description' in prompt_syntax:
                self.description = prompt_syntax['description']
            if 'user_pre_prompt' in prompt_syntax:
                self.user_pre_prompt = prompt_syntax['user_pre_prompt']
            if 'name' in prompt_syntax:
                self.name = prompt_syntax['name']


    def createAgent(self, base_model: Model) -> CustomSystemPromptCodeAgent:
        return CustomSystemPromptCodeAgent(
            tools=[], 
            model=base_model, 
            additional_authorized_imports=["networkx"],
            name=self.name, 
            description=self.description,
            custom_system_prompt=self.additional_system_prompt,
        )