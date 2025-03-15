
from typing import Optional, Dict, Tuple, List

import networkx as nx

from ..factory import AgentFactory
from .causal_inference_agent import StepByStepCausalInferenceAgent
from ...syntax.definitions import Message, InferredVariableDefinition, CausalRelationshipDefinition
from ...utils.message_utils import isInferredVariableDefinition


class BackwardInferenceAgent(StepByStepCausalInferenceAgent):
    """
    Agent running backward to compute exogenous variables from counterfactual outcomes.
    The agent computes counterfactuals following the paradigm from "Backtracking Counterfactuals" [Kügelgen, Mohamed and Beckers , 2023] https://proceedings.mlr.press/v213/kugelgen23a.html
    Note that the agent only computes a single backward pass, to obtain the complete backtracking pipeline, 
    a factual run must be performed and counterfactual runs must go through mimization with factual observations.
    """

    def _prune_graph(self, causal_graph: nx.DiGraph, counterfactual_outcomes: List[Message], observations: List[Message]) -> nx.DiGraph:
        causal_graph = causal_graph.copy()
        for node in counterfactual_outcomes:
            causal_graph.remove_edges_from(list(causal_graph.out_edges(node["name"])))

        kept_nodes = set([node['name'] for node in counterfactual_outcomes])
        conditioning_set = set([observation["name"] for observation in observations])
        for source_node in observations:
            source_name = source_node["name"]
            for target_node in counterfactual_outcomes:
                target_name = target_node["name"]
                paths = self._find_causal_paths(causal_graph, source_name, target_name, conditioning_set - {source_name})
                kept_nodes |= set([n for path in paths for n in path])

        return causal_graph.subgraph(kept_nodes)


    def _run_counterfactual_query(self, causal_graph: nx.DiGraph, counterfactual_outcomes: List[Message], observations: List[Message], run_args: Tuple, run_kwargs: Dict) -> nx.DiGraph:
        # Prune graph
        pruned_graph = self._prune_graph(causal_graph, counterfactual_outcomes, observations)

        # Update counterfactual outcomes
        for counterfactual_outcome in counterfactual_outcomes:
            self._update_node_values_in_place(pruned_graph, counterfactual_outcome, add_causal_effect=True, reset_old_values=False)

        # Reverse graph
        reversed_graph = pruned_graph.reverse()

        # Make predictions
        sinks = [attrs["name"] for node, attrs in reversed_graph.nodes(data=True) if reversed_graph.out_degree(attrs["name"]) == 0]
        updated_causal_graph = reversed_graph
        for target_variable in sinks:
            _, updated_causal_graph = self._estimate_causal_effect(updated_causal_graph, target_variable, run_args, run_kwargs, anticausal=True)

        return updated_causal_graph


    def run(self, *args, additional_args: Optional[Dict] = None, **kwargs):
        if "causal_graph" not in additional_args:
            raise ValueError("Causal graph must be provided as an argument")
        causal_graph = additional_args["causal_graph"]
        
        if "counterfactual_outcomes" not in additional_args:
            raise ValueError("Counterfactual outcomes must be provided as an argument")
        counterfactual_outcomes = additional_args["counterfactual_outcomes"]
        if len(counterfactual_outcomes) == 0:
            raise ValueError("Counterfactual outcomes must be provided as a non-empty list")
        
        if "observations" in additional_args:
            observations = additional_args["observations"]
        else:
            observations = []

        outcome_names = set([outcome["name"] for outcome in counterfactual_outcomes])
        for observation in observations:
            if observation["name"] in outcome_names:
                raise ValueError("Observations must not contain the outcomes")


        for node in causal_graph.nodes:
            if "current_value" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["current_value"]
            if "contextual_information" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["contextual_information"]
            if "causal_effect" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["causal_effect"]

        return self._run_counterfactual_query(causal_graph, counterfactual_outcomes, observations, args, kwargs)




class BackwardInferenceAgentFactory(AgentFactory[BackwardInferenceAgent]):
    
    def __init__(self, path_to_system_prompt: str = 'causal_inference_truncated.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(BackwardInferenceAgent, path_to_system_prompt, use_prompt_lib_folder)

    def createAgent(self, *args, **kwargs) -> BackwardInferenceAgent:
        return super().createAgent(
            *args,
            additional_system_prompt_variables={
                'variable': InferredVariableDefinition.get_definition(),
                'causal_relationship': CausalRelationshipDefinition.get_definition()
            },
            final_answer_checks=[isInferredVariableDefinition],
            **kwargs
        )