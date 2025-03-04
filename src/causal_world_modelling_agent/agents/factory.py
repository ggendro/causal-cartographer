
import yaml
from abc import ABC, abstractmethod


class AgentFactory(ABC):

    def __init__(self, path_to_prompt_syntax: str):
        self.additional_system_prompt = ''
        self.description = ''
        self.user_pre_prompt = ''
        self._readPromptSyntax(path_to_prompt_syntax)

    def _readPromptSyntax(self, path_to_prompt_syntax: str):
        with open(path_to_prompt_syntax, 'r') as file:
            prompt_syntax = yaml.safe_load(file)
            if 'additional_system_prompt' in prompt_syntax:
                self.system_prompt = prompt_syntax['additional_system_prompt']
            if 'description' in prompt_syntax:
                self.description = prompt_syntax['description']
            if 'user_pre_prompt' in prompt_syntax:
                self.user_pre_prompt = prompt_syntax['user_pre_prompt']

    @abstractmethod
    def createAgent(self):
        pass