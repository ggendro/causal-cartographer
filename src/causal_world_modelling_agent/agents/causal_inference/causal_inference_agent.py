
from typing import Optional, Dict, Tuple, List

import networkx as nx

from ..factory import AgentFactory
from ..custom_prompt_agent import CustomPromptAgent
from ...syntax.definitions import Message, InferredVariableDefinition, CausalRelationshipDefinition
from ...utils.message_utils import isInferredVariableDefinition


class StepByStepCausalInferenceAgent(CustomPromptAgent):

    def __init__(self, *args, traversal_cutoff: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.traversal_cutoff = traversal_cutoff

    
    def _reset_state(self) -> None: # Needed to make each run independent
        self.python_executor.state = {}


    def _update_node_values_in_place(self, causal_graph: nx.DiGraph, updates: Message, add_causal_effect: bool = True, reset_old_values: bool = False) -> None:
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
    

    def _build_intervened_graph(self, causal_graph: nx.DiGraph, interventions: List[Message], structure_only: bool = False) -> nx.DiGraph:
        intervened_graph = causal_graph.copy()
        for intervention in interventions:
            if not structure_only:
                self._update_node_values_in_place(intervened_graph, intervention)
            intervened_graph.remove_edges_from(list(intervened_graph.in_edges(intervention["name"])))
        return intervened_graph
    

    def _path_to_chain(self, path: List[str], graph: nx.DiGraph) -> nx.DiGraph:
        chain = nx.DiGraph()
        chain.add_nodes_from(path)

        for a, b in zip(path[:-1], path[1:]):
            if graph.has_edge(a, b):
                chain.add_edge(a, b)
            elif graph.has_edge(b, a):
                chain.add_edge(b, a)
            else:
                raise ValueError("Path does not exist in graph")
        
        return chain

    def _find_causal_paths(self, causal_graph: nx.DiGraph, source: str, target: str, conditioning_set: List[str]) -> List[List[str]]:
        candidate_paths = nx.all_simple_paths(causal_graph.to_undirected(as_view=True), source=source, target=target, cutoff=self.traversal_cutoff)

        causal_paths = []
        for path in candidate_paths:
            chain = self._path_to_chain(path, causal_graph)
            conditioning_subset = set(conditioning_set) & set(chain.nodes)
            if not nx.is_d_separator(chain, source, target, conditioning_subset):
                causal_paths.append(path)

        return causal_paths


    def _prune_graph(self, causal_graph: nx.DiGraph, target_variable: str, observations: List[Message], interventions: List[Message]) -> nx.DiGraph:
        intervened_graph = self._build_intervened_graph(causal_graph, interventions, structure_only=True)

        # Add nodes that are both observed and intervened on using the twin graph method
        obs_and_inter_nodes = set([observation["name"] for observation in observations]) & set([intervention["name"] for intervention in interventions])
        obs_and_inter_nodes = {f"obs_and_inter_node_{hash(node)}": node for node in obs_and_inter_nodes}
        for node in obs_and_inter_nodes.keys():
            intervened_graph.add_node(node, name=node)
            matching_in_edges = list(causal_graph.in_edges(obs_and_inter_nodes[node]))
            for source, _ in matching_in_edges:
                intervened_graph.add_edge(source, node)

        intervened_graph.remove_edges_from(list(intervened_graph.out_edges(target_variable))) # key aspect: we compute the causal effect from parent knowledge only

        kept_nodes = set([target_variable])
        conditioning_set = set([observation["name"] for observation in observations] + [intervention["name"] for intervention in interventions] + list(obs_and_inter_nodes.keys()))
        for node_name in conditioning_set:
            paths = self._find_causal_paths(intervened_graph, node_name, target_variable, conditioning_set - {node_name})
            kept_nodes |= set([n for path in paths for n in path])

        # Update twin nodes with their real values
        for node in obs_and_inter_nodes.keys():
            if node in kept_nodes:
                kept_nodes.remove(node)
                kept_nodes.add(obs_and_inter_nodes[node])

        return causal_graph.subgraph(kept_nodes).copy()


    def _estimate_causal_effect_in_place(self, causal_graph: nx.DiGraph, target_variable: str, run_args: Tuple, run_kwargs: Dict, anticausal: bool = False) -> None:
        if "causal_effect" not in causal_graph.nodes[target_variable]:
            parents = list(causal_graph.predecessors(target_variable)) # key aspect: we compute the causal effect from parent knowledge only
            for parent in parents:
                self._estimate_causal_effect_in_place(causal_graph, parent, run_args, run_kwargs, anticausal=anticausal)

            if not anticausal:
                parent_name = 'parent_variables'
            else:
                parent_name = 'children_variables'
                
            updated_variable = super().run(*run_args, additional_args={parent_name: [causal_graph.nodes[parent] for parent in parents], 'target_variable': causal_graph.nodes[target_variable], 'causal_relationships': list(causal_graph.in_edges(target_variable, data=True))}, **run_kwargs)
            self._reset_state()
            self._update_node_values_in_place(causal_graph, updated_variable)

    def _estimate_causal_effect(self, causal_graph: nx.DiGraph, target_variable: str, run_args: Tuple, run_kwargs: Dict, anticausal: bool = False) -> Tuple[str, nx.DiGraph]:
        causal_graph = causal_graph.copy()
        self._estimate_causal_effect_in_place(causal_graph, target_variable, run_args,	run_kwargs, anticausal=anticausal)

        return causal_graph.nodes[target_variable]["causal_effect"], causal_graph
    

    def _compute_abductions(self, causal_graph: nx.DiGraph, observations: List[Message], interventions: List[Message], run_args: Tuple, run_kwargs: Dict) -> List[Message]:
        causal_graph = causal_graph.copy()

        non_abduction_nodes = [observation["name"] for observation in observations] + [intervention["name"] for intervention in interventions]
        abduction_nodes = []
        for node in causal_graph.nodes:
            if causal_graph.in_degree(node) == 0 and node not in non_abduction_nodes:
                abduction_nodes.append(node)

        for observation in observations:
            self._update_node_values_in_place(causal_graph, observation)
        
        abductions = []
        for node in abduction_nodes:
            reverse_graph = causal_graph.reverse(copy=True)
            pruned_reverse_graph = self._prune_graph(reverse_graph, node, observations, [])
            
            # Perform an initial forward causal pass to compute values of colliders (i.e. nodes with no incoming edges in the reverse graph) and reduce the relaince on anticausal reasoning
            for collider in pruned_reverse_graph.nodes:
                if node != collider and pruned_reverse_graph.in_degree(collider) == 0 and "causal_effect" not in pruned_reverse_graph.nodes[collider]:
                    subgraph = self._prune_graph(causal_graph, collider, [], observations) # treat observations as interventions to consider only direct causal effects, block backdoor paths and avoid infinite loops
                    _, updated_subgraph = self._estimate_causal_effect(subgraph, collider, run_args, run_kwargs)
                    for updated_node in updated_subgraph.nodes:
                        if "causal_effect" in updated_subgraph.nodes[updated_node]:
                            self._update_node_values_in_place(pruned_reverse_graph, updated_subgraph.nodes[updated_node])

            # Compute values of abduction nodes
            _, updated_graph = self._estimate_causal_effect(pruned_reverse_graph, node, run_args, run_kwargs, anticausal=True)
            abductions.append(updated_graph.nodes[node])

        return abductions
    

    def _run_counterfactual_query(self, causal_graph: nx.DiGraph, target_variable: str, observations: List[Message], interventions: List[Message], run_args: Tuple, run_kwargs: Dict) -> Tuple[str, nx.DiGraph]:
        # Prune graph
        pruned_graph = self._prune_graph(causal_graph, target_variable, observations, interventions)

        # Remove pruned nodes from observations
        for i in range(len(observations) - 1, -1, -1):
            if observations[i]["name"] not in pruned_graph.nodes:
                del observations[i]

        # Remove pruned nodes from interventions
        for i in range(len(interventions) - 1, -1, -1):
            if interventions[i]["name"] not in pruned_graph.nodes:
                del interventions[i]        

        # Compute abductions
        abductions = self._compute_abductions(pruned_graph, observations, interventions, run_args, run_kwargs)
        for abduction in abductions:
            self._update_node_values_in_place(pruned_graph, abduction)

        # Intervene on the causal graph
        intervened_graph = self._build_intervened_graph(pruned_graph, interventions, structure_only=False)

        # Add observations that are not affected by the interventions
        intervention_names = set([intervention["name"] for intervention in interventions])
        for observation in observations:
            observation_name = observation["name"]
            if observation_name not in intervention_names and intervened_graph.in_degree(observation_name) == 0: # TODO: verify that condition is correct
                self._update_node_values_in_place(intervened_graph, observation)

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

        if target_variable not in causal_graph.nodes:
            raise ValueError("Target variable not found in causal graph")
        
        if target_variable in [observation["name"] for observation in observations]:
            raise ValueError("Target variable cannot be observed")
        
        if target_variable in [intervention["name"] for intervention in interventions]:
            raise ValueError("Target variable cannot be intervened on")

        for node in causal_graph.nodes:
            if "current_value" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["current_value"]
            if "contextual_information" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["contextual_information"]
            if "causal_effect" in causal_graph.nodes[node]:
                del causal_graph.nodes[node]["causal_effect"]

        return self._run_counterfactual_query(causal_graph, target_variable, observations, interventions, args, kwargs)




class CausalInferenceAgentFactory(AgentFactory[StepByStepCausalInferenceAgent]):

    def __init__(self, path_to_system_prompt: str = 'causal_inference_truncated.yaml', use_prompt_lib_folder: bool = True):
        super().__init__(StepByStepCausalInferenceAgent, path_to_system_prompt, use_prompt_lib_folder)

    def createAgent(self, *args, **kwargs) -> StepByStepCausalInferenceAgent:
        return super().createAgent(
            *args,
            additional_system_prompt_variables={
                'variable': InferredVariableDefinition.get_definition(),
                'causal_relationship': CausalRelationshipDefinition.get_definition()
            },
            final_answer_checks=[isInferredVariableDefinition],
            **kwargs
        )