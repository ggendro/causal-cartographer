
from typing import Optional

from smolagents import CodeAgent




class CustomSystemPromptCodeAgent(CodeAgent):

    def __init__(self, *args, custom_system_prompt: Optional[str] = None, replace_existing_system_prompt_with_custom: bool = False, prefix_cutom_system_prompt: bool = True, **kwargs):
        self.custom_system_prompt = custom_system_prompt
        self.replace_existing_system_prompt_with_custom = replace_existing_system_prompt_with_custom
        self.prefix_cutom_system_prompt = prefix_cutom_system_prompt

        super().__init__(*args, **kwargs)

    def initialize_system_prompt(self):
        system_prompt = super().initialize_system_prompt()

        if self.custom_system_prompt:
            if self.replace_existing_system_prompt_with_custom:
                system_prompt = self.custom_system_prompt
            elif self.prefix_cutom_system_prompt:
                system_prompt = self.custom_system_prompt + '\n\nIn addition: ' + system_prompt
            else:
                system_prompt += '\n\nIn addition: ' + self.custom_system_prompt

        return system_prompt