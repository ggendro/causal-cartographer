
from typing import Optional, Dict, Tuple, List

import networkx as nx
from smolagents import Model

from ..factory import AgentFactory
from ...core.agent import CustomSystemPromptCodeAgent


class StepByStepCausalInferenceAgent(CustomSystemPromptCodeAgent):

    def __init__(self, *args, traversal_cutoff: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.traversal_cutoff = traversal_cutoff


    def _update_node_values_in_place(self, causal_graph: nx.DiGraph, updates: Dict[str, str], add_causal_effect: bool = True, reset_old_values: bool = False) -> None:
        node_name = updates["name"]

        if reset_old_values:
            for key in causal_graph.nodes[node_name]:
                if key != "name":
                    del causal_graph.nodes[node_name][key]

        for key, value in updates.items():
            if key != "name":
                causal_graph.nodes[node_name][key] = value

        if add_causal_effect and "current_value" in updates and "causal_effect" not in updates:
            causal_graph.nodes[node_name]["causal_effect"] = updates["current_value"]
    

    def _build_intervened_graph(self, causal_graph: nx.DiGraph, interventions: List[Dict[str, str]], structure_only: bool = False) -> nx.DiGraph:
        intervened_graph = causal_graph.copy()
        for intervention in interventions:
            if not structure_only:
                self._update_node_values_in_place(intervened_graph, intervention, add_causal_effect=True, reset_old_values=False)
            intervened_graph.remove_edges_from(list(intervened_graph.in_edges(intervention["name"])))
        return intervened_graph
    
    
    def _find_causal_paths(self, causal_graph: nx.DiGraph, source: str, target: str, conditioning_set: List[str]) -> List[List[str]]:
        candidate_paths = nx.all_simple_paths(causal_graph.to_undirected(as_view=True), source=source, target=target, cutoff=self.traversal_cutoff)

        causal_paths = []
        for path in candidate_paths:
            path = causal_graph.subgraph(path)
            conditioning_subset = set(conditioning_set) & set(path.nodes)
            if not nx.is_d_separator(path, source, target, conditioning_subset):
                causal_paths.append(path)

        return causal_paths


    def _prune_graph(self, causal_graph: nx.DiGraph, target_variable: str, observations: List[Dict[str, str]], interventions: List[Dict[str, str]]) -> nx.DiGraph:
        intervened_graph = self._build_intervened_graph(causal_graph, interventions, structure_only=True)
        intervened_graph.remove_edges_from(list(intervened_graph.out_edges(target_variable)))

        kept_nodes = set()
        conditioning_set = set([observation["name"] for observation in observations] + [intervention["name"] for intervention in interventions])
        for node in observations + interventions:
            node_name = node["name"]
            paths = self._find_causal_paths(intervened_graph, node_name, target_variable, conditioning_set - {node_name})
            kept_nodes |= set([n for path in paths for n in path])
        
        kept_nodes |= set(intervened_graph.predecessors(target_variable))

        return causal_graph.subgraph(kept_nodes).copy()


    def _estimate_causal_effect_in_place(self, causal_graph: nx.DiGraph, target_variable: str, run_args: Tuple, run_kwargs: Dict) -> None:
        if "causal_effect" not in causal_graph.nodes[target_variable]:
            parents = list(causal_graph.predecessors(target_variable))
            for parent in parents:
                self._estimate_causal_effect_in_place(causal_graph, parent, run_args, run_kwargs)
                
            updated_variable = super().run(*run_args, additional_args={'parent_variables': [causal_graph.nodes[parent] for parent in parents], 'target_variable': causal_graph.nodes[target_variable], 'causal_relationships': causal_graph.in_edges(target_variable, data=True)}, **run_kwargs)
            self._update_node_values_in_place(causal_graph, updated_variable, add_causal_effect=True, reset_old_values=False)

    def _estimate_causal_effect(self, causal_graph: nx.DiGraph, target_variable: str, run_args: Tuple, run_kwargs: Dict) -> Tuple[str, nx.DiGraph]:
        causal_graph = causal_graph.copy()
        self._estimate_causal_effect_in_place(causal_graph, target_variable, run_args,	run_kwargs)

        return causal_graph.nodes[target_variable]["causal_effect"], causal_graph
    

    def _compute_abductions(self, causal_graph: nx.DiGraph, observations: List[Dict[str, str]], interventions: List[Dict[str, str]], run_args: Tuple, run_kwargs: Dict) -> List[Dict[str, str]]:
        non_abduction_nodes = [observation["name"] for observation in observations] + [intervention["name"] for intervention in interventions]
        abduction_nodes = []
        for node in causal_graph.nodes:
            if causal_graph.in_degree(node) == 0 and node not in non_abduction_nodes:
                abduction_nodes.append(node)
        
        reverse_graph = causal_graph.reverse()

        for observation in observations:
            self._update_node_values_in_place(reverse_graph, observation, add_causal_effect=True, reset_old_values=False)

        abductions = []
        for node in abduction_nodes:
            _, updated_graph = self._estimate_causal_effect(reverse_graph, node, run_args, run_kwargs)
            abductions.append(updated_graph.nodes[node])

        return abductions
    

    def _run_counterfactual_query(self, causal_graph: nx.DiGraph, target_variable: str, observations: List[Dict[str, str]], interventions: List[Dict[str, str]], run_args: Tuple, run_kwargs: Dict) -> Tuple[str, nx.DiGraph]:
        # Prune graph
        pruned_graph = self._prune_graph(causal_graph, target_variable, observations, interventions)

        # Compute abductions
        abductions = self._compute_abductions(pruned_graph, observations, interventions, run_args, run_kwargs)
        for abduction in abductions:
            self._update_node_values_in_place(pruned_graph, abduction, add_causal_effect=True, reset_old_values=False)

        # Intervene on the causal graph
        intervened_graph = self._build_intervened_graph(pruned_graph, interventions, structure_only=False)

        # Add observations that are not affected by the interventions
        for observation in observations:
            observation_name = observation["name"]
            if intervened_graph.in_degree(observation_name) == 0: # TODO: verify that condition is correct
                self._update_node_values_in_place(intervened_graph, observation, add_causal_effect=True, reset_old_values=False)

        # Make predictions
        causal_effect, updated_causal_graph = self._estimate_causal_effect(intervened_graph, target_variable, run_args, run_kwargs)

        return causal_effect, updated_causal_graph


    def run(self, *args, additional_args: Optional[Dict] = None, **kwargs):
        if "causal_graph" not in additional_args:
            raise ValueError("Causal graph must be provided as an argument")
        causal_graph = additional_args["causal_graph"]
        
        if "target_variable" not in additional_args:
            raise ValueError("Target variable must be provided as an argument")
        target_variable = additional_args["target_variable"]
        
        if "observations" not in additional_args:
            observations = []
        else:
            observations = additional_args["observations"]
        
        if "interventions" not in additional_args:
            interventions = []
        else:
            interventions = additional_args["interventions"]


        for node in causal_graph.nodes:
            if "current_value" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["current_value"]
            if "contextual_information" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["contextual_information"]
            if "causal_effect" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["causal_effect"]

        return self._run_counterfactual_query(causal_graph, target_variable, observations, interventions, args, kwargs)




class CausalInferenceAgentFactory(AgentFactory):

    DESCRIPTION = """Agent that computes a causal inference query given a causal graph and observational, interventional and counterfactual data. All the additional arguments must be provided as a dictionary with the following keys 'causal_graph', 'target_variable', 'observations' and 'interventions'.
                    The causal graph is provided as a networkx DiGraph. Observations are provided as a list of dictionaries corresponding to the observed variables and their values. Interventions are provided as a list of dictionaries with the intervened variables and their values.
                    The agent returns a tuple with the causal effect as a string and the updated networkx causal graph.
                    """
    
    SYSTEM_PROMPT = """You are an agent that computes the causal effect of a target causal variable given the values of its direct parents in the causal graph. The agent must use common sense knowledge to estimate the causal effect of the target variable.
                    The attributes of the target variable are provided as arguments with the name 'target_variable'. The parent variables attributes are provided as a list of dictionaries with the name 'parent_variables'. The descriptions of the causal relationships between the target varibale and its parents are provided as a list of attribute dictionaries with the name 'causal_relationships'.
                    The agent must return a dictionary with the updated attributes of the target variable. The fields to update are 'current_value', 'contextual_information' and 'causal_effect'.
                    
                    Example:
                    ---
                    Task: [USER_INPUT]

                    Thought: Let's print out the variable attributes: 
                    ```
                    print("target variable:", target_variable)
                    for i, parent_variable in enumerate(parent_variables):
                        print(f"parent variable {i}: {parent_variable}")
                    ```<end_code>
                    Observation: The target variable is '...' and the parent variables are '...'. They are linked by the following causal relationships: '...'. By consequence, the causal effect is '...'.
                    
                    target_variable['current_value'] = '...'
                    target_variable['contextual_information'] = '...'
                    target_variable['causal_effect'] = '...'
                    final_answer(target_variable)
                    ```<end_code>
                    """

    def createAgent(self, base_model: Model) -> StepByStepCausalInferenceAgent:
        return StepByStepCausalInferenceAgent(
                    tools=[],
                    model=base_model, 
                    additional_authorized_imports=[],
                    name="causal_inference_agent", 
                    description=CausalInferenceAgentFactory.DESCRIPTION,
                    custom_system_prompt=CausalInferenceAgentFactory.SYSTEM_PROMPT,
                    managed_agents=[]
        )