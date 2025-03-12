
from typing import Dict, Optional

from smolagents import CodeAgent
from smolagents.agents import populate_template




class CustomPromptAgent(CodeAgent):

    def __init__(self, *args, additional_system_prompt_variables: Optional[Dict[str, str]] = None, **kwargs):
        if not additional_system_prompt_variables:
            additional_system_prompt_variables = {}
        self.additional_system_prompt_variables = additional_system_prompt_variables

        super().__init__(*args, **kwargs)

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "authorized_imports": (
                    "You can import from any package you want."
                    if "*" in self.authorized_imports
                    else str(self.authorized_imports)
                ),
                **self.additional_system_prompt_variables,
            },
        )
        return system_prompt