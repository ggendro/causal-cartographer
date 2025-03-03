
from typing import Optional, Dict

from smolagents import Model

from ..factory import AgentFactory
from ...core.agent import CustomSystemPromptCodeAgent
# from .causal_inference_agent import CausalInferenceAgentFactory
# from .counterfactual_generator_agent import CounterfactualGeneratorAgentFactory
# from .counterfactual_evaluator_agent import CounterfactualEvaluatorAgentFactory
# from .graph_evaluator_agent import GraphEvaluatorAgentFactory
# from ...evaluators.evaluators import mixup_evaluator
# from ...evaluators.evaluators import matching_evaluator
# from ...evaluators.evaluators import complexity_evaluator
# from ...evaluators.evaluators import bound_targeting_evaluator
# from ...evaluators.evaluators import entropy_evaluator 


class CausalInferenceManagerAgent(CustomSystemPromptCodeAgent):

    def run(self, *args, additional_args: Optional[Dict] = None, **kwargs):
        if "causal_graph" not in additional_args:
            raise ValueError("Causal graph must be provided as an argument")
        
        return super().run(*args, additional_args=additional_args, **kwargs)


class CausalInferenceManagerAgentFactory(AgentFactory):

    DESCRIPTION = """
                    Agent that manages other agents to execute causal observation, intervention and counterfactual reasoning tasks and evaluate the ability of the causal graph to fit the data.
                    """

    def createAgent(self, base_model: Model) -> CausalInferenceManagerAgent:
        # causal_inference_agent = CausalInferenceAgentFactory().createAgent(base_model)
        # counterfactual_generator_agent = CounterfactualGeneratorAgentFactory().createAgent(base_model)
        # counterfactual_evaluator_agent = CounterfactualEvaluatorAgentFactory().createAgent(base_model)
        # graph_evaluator_agent = GraphEvaluatorAgentFactory().createAgent(base_model)

        return CausalInferenceManagerAgent(
                    # tools=[mixup_evaluator, matching_evaluator, complexity_evaluator, bound_targeting_evaluator, entropy_evaluator],
                    model=base_model, 
                    additional_authorized_imports=[],
                    name="causal_inference_agent", 
                    description=CausalInferenceManagerAgentFactory.DESCRIPTION,
                    # managed_agents=[causal_inference_agent, counterfactual_generator_agent, counterfactual_evaluator_agent, graph_evaluator_agent]
        )