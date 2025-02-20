
from smolagents import CodeAgent, Model

from ..factory import AgentFactory


class CausalInferenceAgentFactory(AgentFactory):

    def createAgent(self, base_model: Model) -> CodeAgent:
        return CodeAgent(
                    tools=[], 
                    model=base_model, 
                    additional_authorized_imports=[],
                    name="causal_inference_agent", 
                    description="""
                                
                                """
        )