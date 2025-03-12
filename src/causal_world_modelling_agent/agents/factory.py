
from typing import Optional, TypeVar, Generic
import yaml
import os

from smolagents import Model, CodeAgent



def toPromptFolder() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'syntax', 'prompts')


T = TypeVar('T', bound=CodeAgent)
class AgentFactory(Generic[T]):

    def __init__(self, agent_type: type[CodeAgent] = CodeAgent, path_to_system_prompt: Optional[str] = None, use_prompt_lib_folder: bool = True):
        self.agent_type = agent_type

        self.description = ''
        self.user_pre_prompt = ''
        self.name = ''
        self.prompt_templates =None

        if path_to_system_prompt:
            if use_prompt_lib_folder:
                path_to_system_prompt = os.path.join(toPromptFolder(), path_to_system_prompt)

            self._readPromptSyntax(path_to_system_prompt)

    def _readPromptSyntax(self, path_to_system_prompt: str) -> None:
        with open(path_to_system_prompt, 'r') as file:
            prompt_syntax = yaml.safe_load(file)

            if 'description' in prompt_syntax:
                self.description = prompt_syntax['description']
                del prompt_syntax['description']
            if 'user_pre_prompt' in prompt_syntax:
                self.user_pre_prompt = prompt_syntax['user_pre_prompt']
                del prompt_syntax['user_pre_prompt']
            if 'name' in prompt_syntax:
                self.name = prompt_syntax['name']
                del prompt_syntax['name']

            self.prompt_templates = None
            if path_to_system_prompt != 'code_agent.yaml': # Default prompt template
                with open(os.path.join(toPromptFolder(), 'code_agent.yaml'), 'r') as default_templates_file:
                    prompt_templates = yaml.safe_load(default_templates_file)
                    prompt_templates.update(prompt_syntax)
                    self.prompt_templates = prompt_templates


    def createAgent(self, base_model: Model, tools: Optional[list] = None, *args, **kwargs) -> T:
        if tools is None:
            tools = []

        return self.agent_type(
            *args,
            tools=tools, 
            model=base_model,
            prompt_templates=self.prompt_templates,
            additional_authorized_imports=["networkx"],
            name=self.name, 
            description=self.description,
            **kwargs
        )