
import networkx as nx
from typing import Optional, List, Dict, Generator, Optional, Set, Callable, Tuple
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
import itertools
import math

from ..syntax.definitions import Message, VariableDefinition, _CausalEffectDefinition
from ..utils.inference_utils import build_target_node_causal_blanket, dagify, find_causal_paths, build_intervened_graph
from ..tools.evaluators import observation_constrained_causal_graph_entropy



@dataclass
class Query:
    world_ids: List[str]
    causal_graph: nx.DiGraph
    target_node: str
    ground_truth: str
    is_pseudo_gt: bool = False
    observations: Optional[List[Message]] = None
    interventions: Optional[List[Message]] = None



class _WorldManager(ABC):

    def __init__(self, initial_graph: Optional[nx.DiGraph] = None, graphs: Optional[List[nx.DiGraph]] = None, traversal_cutoff: Optional[int] = None):
        self.traversal_cutoff = traversal_cutoff

        if initial_graph is None:
            self.graph = nx.DiGraph()
        else:
            self.graph = initial_graph.copy()

        if graphs is not None:
            for graph in graphs:
                self.merge(graph)
        
    def get_complete_graph(self) -> nx.DiGraph:
        return self.graph
    
    def get_traversal_cutoff(self) -> Optional[int]:
        return self.traversal_cutoff
    
    def set_traversal_cutoff(self, traversal_cutoff: Optional[int]) -> None:
        self.traversal_cutoff = traversal_cutoff

    @abstractmethod
    def merge(self, new_graph: nx.DiGraph, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def get_worlds(self) -> Dict[str, nx.DiGraph]:
        pass

    @abstractmethod
    def get_world(self, world_id: str) -> nx.DiGraph:
        pass

    @abstractmethod
    def get_worlds_from_node(self, target_node: str) -> Dict[str, nx.DiGraph]:
        pass

    @abstractmethod
    def match_counterfactuals(self, target_node: str, num_interventions: int = 1) -> Generator[Query, None, None]:
        pass

    @abstractmethod
    def mix_counterfactuals(self, target_node: str, num_interventions: int = 1) -> Generator[Query, None, None]:
        pass




def extract_world_nodes(graph: nx.DiGraph, key_syntax) -> Dict[str, Set[str]]:
    """
    Extract nodes from a directed graph based on matching attribute keys.
    This function iterates over each node in the provided NetworkX directed graph and
    checks each attribute key against the given regular expression syntax. When a key matches,
    the matching portion of the key is used as the world identifier, and the node is added
    to the set associated with that identifier.
    Parameters:
        graph (nx.DiGraph): A directed graph whose nodes contain attribute dictionaries.
        key_syntax (str or Pattern): A regular expression (or compiled regex pattern) used to
            match attribute keys. The portion of the key that matches is treated as a world ID.
    Returns:
        Dict[str, Set[str]]: A dictionary mapping each world ID (derived from matching attribute keys)
            to a set of node identifiers that contain a key matching the provided syntax.
    """
    worlds = {}
    for node, attrs in graph.nodes(data=True):
        for key in attrs.keys():
            key_match = re.fullmatch(key_syntax, key)
            if key_match is not None:
                world_id = key_match.group(0)
                if world_id not in worlds:
                    worlds[world_id] = set()
                worlds[world_id].add(node)

    return worlds


def ground_in_world(graph: nx.DiGraph, world_id: str | None, key_syntax: str) -> nx.DiGraph:
    """
    Remove node attribute keys that do not match the specified world_id.
    This function iterates over all nodes in the graph and inspects each attribute key. If a key matches the provided regular expression pattern (key_syntax)
    but its matched group (group(0)) is not equal to the specified world_id, the key is removed from the node's attributes. The moethod returns a copy of the graph with the specified keys removed.
    Parameters:
        graph (nx.DiGraph): The input directed graph whose nodes may contain attribute keys.
        world_id (str | None): The identifier used to determine which attribute keys to preserve. Keys with a matching group equal to this identifier are kept. If None, all keys matching the key_syntax are removed.
        key_syntax (str): A regular expression pattern used to match keys in the node attributes.
    Returns:
        nx.DiGraph: A modified copy of the input graph with attribute keys removed according to the specified criteria.
    """
    graph = graph.copy()
    for _, attrs in graph.nodes(data=True):
        for key in list(attrs.keys()):
            key_match = re.fullmatch(key_syntax, key)
            if key_match is not None:
                if key_match.group(0) != world_id:
                    del attrs[key]

    return graph


def find_shared_observations(observations_1: Dict[str, Message], observations_2: Dict[str, Message]) -> Dict[str, Message]:
    """
    Finds and returns the shared observations between two dictionaries of messages.
    This function compares two dictionaries where each key represents a node's name and each value is a Message
    that includes the 'current_value' key. It returns a new dictionary containing only those messages from the first
    dictionary that are also present in the second dictionary and have matching 'current_value'.
    Parameters:
        observations_1 (Dict[str, Message]): A dictionary containing messages identified by node names. Each message
            must contain a 'current_value' field.
        observations_2 (Dict[str, Message]): Another dictionary with messages, keyed by node names, to be compared
            against the first dictionary.
    Returns:
        Dict[str, Message]: A dictionary of messages from observations_1 that exist in observations_2 with identical
            'current_value' fields.
    """
    shared_observations = {}
    for node_name in observations_1:
        if node_name in observations_2 and observations_1[node_name]['current_value'] == observations_2[node_name]['current_value']:
            shared_observations[node_name] = observations_1[node_name]

    return shared_observations


def find_active_interventions(causal_graph: nx.DiGraph, target_node: str, observations: Dict[str, Message], interventions: Dict[str, Message], traversal_cutoff: Optional[int] = None) -> Dict[str, Message]:    
    """
    Find active interventions that influence a target node in a causal graph.
    This function inspects each intervention (excluding the target node itself) and determines if there exists at least
    one causal path from the intervention node to the target node within the provided causal graph. Paths are determined
    using observed nodes as a conditioning set and can be bounded by a specified traversal cutoff.
    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing causal relationships among nodes.
        target_node (str): The node for which the active interventions are being identified.
        observations (Dict[str, Message]): A dictionary mapping observed node names to their corresponding messages. Observations are used as a conditioning set for pathfinding.
        interventions (Dict[str, Message]): A dictionary mapping intervention node names to their intervention messages. The returned dictionary is a subset of this one, containing only those interventions that have at least one non-blocked causal path to the target node.
        traversal_cutoff (Optional[int]): Optional maximum depth for searching causal paths; if provided, limits the traversal.
    Returns:
        Dict[str, Message]: A dictionary containing only those interventions that have at least one causal path to the target node.
    """
    active_interventions = {}

    if target_node not in causal_graph.nodes:
        raise ValueError(f'Target node {target_node} not found in the causal graph')
    
    if target_node in observations:
        return active_interventions # No active interventions if target node is in observations
    
    if target_node in interventions:
        active_interventions[target_node] = interventions[target_node] # Include the target node as an active intervention if it is in the interventions dictionary
        return active_interventions

    causal_graph = build_intervened_graph(causal_graph, interventions.values(), structure_only=True) # Build a new graph with the interventions applied
    for intervention_node, intervention_value in interventions.items():
        if intervention_node != target_node:
            causal_paths = find_causal_paths(causal_graph, intervention_node, target_node, list(observations.keys() - {intervention_node}), traversal_cutoff=traversal_cutoff)
            if len(causal_paths) > 0:
                active_interventions[intervention_node] = intervention_value

    return active_interventions


def find_non_blocking_observations(causal_graph: nx.DiGraph, target_node: str, observations: Dict[str, Message], blocking_observations: Dict[str, Message], interventions: Dict[str, Message], traversal_cutoff: Optional[int] = None) -> Dict[str, Message]:
    """
    Find observations that influence the target node while not blocking interventions and being blocked by the blocking observations.
    This function inspects each observation and determines if there exists at least one causal path from the observation node to the target node within the provided causal graph. 
    Paths are determined using observed nodes and can be bounded by a specified traversal cutoff. Interventions act as blocking nodes. Conversely, at least one blocking observation must be on the path.
    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing causal relationships among nodes.
        target_node (str): The node for which the non-blocking observations are being identified.
        observations (Dict[str, Message]): A dictionary mapping observed node names to their corresponding messages. The returned dictionary is a subset of this one, containing only those observations that have at least one causal path to the target node respecting the conditions.
        blocking_observations (Dict[str, Message]): A dictionary mapping blocking node names to their corresponding messages. Blocking observations must be on the path to the target node.
        interventions (Dict[str, Message]): A dictionary mapping intervention node names to their intervention messages. Interventions are used as blocking nodes in the pathfinding process.
        traversal_cutoff (Optional[int]): Optional maximum depth for searching causal paths; if provided, limits the traversal.
    Returns:
        Dict[str, Message]: A dictionary containing only those observations that have at least one causal path to the target node and are not blocked by the blocking observations.
    """
    intervened_graph = build_intervened_graph(causal_graph, interventions.values(), structure_only=True)
    non_blocking_observations = {}
    for observation_node, observation_value in observations.items():
        if observation_node != target_node:
            blocked_paths = find_causal_paths(intervened_graph, observation_node, target_node, list(blocking_observations.keys()), traversal_cutoff=traversal_cutoff) # Verify that the path is blocked by the blocking observations
            if len(blocked_paths) == 0:
                causal_paths = find_causal_paths(causal_graph, observation_node, target_node, list(interventions.keys()), traversal_cutoff=traversal_cutoff) # Verify that the path is not blocked by the interventions
                if len(causal_paths) > 0:
                    non_blocking_observations[observation_node] = observation_value

    return non_blocking_observations



def find_mixing_coefficient_linear(
        causal_graph: nx.DiGraph, 
        target_node: str, 
        observations_1: Dict[str, Message], 
        observations_2: Dict[str, Message], 
        traversal_cutoff: Optional[int] = None, 
        epsilon: float = 0.25,
        factual_target_value: Optional[str] = None,
        counterfactual_target_value: Optional[str] = None,
        **kwargs) -> Dict[str,float]:
    """
    Find the mixing coefficient between two sets of observations in a causal graph using linear combination.
    This function inspects the causal graph and determines the mixing coefficient between the expected outputs from two sets of observations. Only the outputs from the observed worlds are considered.
    The mixing coefficient is defined as the average of the coefficients of the paths from the target node to the observations.
    The coefficient is calculated recursively, and the traversal can be limited by a specified cutoff depth.
    A higher mixing coefficient indicates a stronger influence of the set of observations 1 over the set of observations 2.
    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing causal relationships among nodes.
        target_node (str): The node for which the mixing coefficient is being calculated.
        observations_1 (Dict[str, Message]): A dictionary mapping node names to their corresponding messages representing the first set of observations.
        observations_2  (Dict[str, Message]): A dictionary mapping node names to their corresponding messages representing the second set of observations.
        traversal_cutoff (Optional[int]): Optional maximum depth for searching causal paths; if provided, limits the traversal.
        epsilon (float): A small positive value used to control the sensitivity of the mixing coefficient calculation. It should be between 0 and 1 (exclusive).
        factual_target_value (Optional[str]): The factual target value for the target node. If not provided, it is deduced from the observations.
        counterfactual_target_value (Optional[str]): The counterfactual target value for the target node. If not provided, it is deduced from the observations.
    Returns:
        float: The mixing coefficient between the two sets of observations with respect to the target node.
    """
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError('Epsilon must be between 0 and 1')
    
    coeffs = {}
    def _find_mix_rec(node: str, depth: int) -> float:
        if node in coeffs:
            return coeffs[node]
        if node in observations_1:
            return 1.0
        if node in observations_2:
            return -1.0
        if traversal_cutoff is not None and depth > traversal_cutoff:
            return 0.0
        
        parents = list(causal_graph.predecessors(node))
        if len(parents) == 0:
            return 0.0
        
        coeff = sum([_find_mix_rec(parent, depth + 1) for parent in parents]) / len(parents) # normalize the coefficient by the number of parents
        coeffs[node] = coeff
        return coeff
    
    coeff = _find_mix_rec(target_node, 0)
    coeff = 1 / (1 + math.exp(-coeff / epsilon)) # sigmoid function to normalize the coefficient between 0 and 1

    if not factual_target_value: # Observing the target node may lead to inconsistencies in the ground truth. TODO: assess if it should be allowed
        factual_target_value = observations_1[target_node]['current_value']
    if not counterfactual_target_value: 
        counterfactual_target_value = observations_2[target_node]['current_value']

    ground_truth = {
        factual_target_value: coeff,
        counterfactual_target_value: 1 - coeff
    }
    return ground_truth



def find_mixing_coefficient_deviation_intervals(
        causal_graph: nx.DiGraph, 
        target_node: str, 
        observations_1: Dict[str, Message], 
        observations_2: Dict[str, Message], 
        traversal_cutoff: Optional[int] = None, 
        epsilon: float = 0.25, 
        world_set: Optional[List[nx.DiGraph]] = None,
        factual_target_value: Optional[str] = None,
        counterfactual_target_value: Optional[str] = None,
        **kwargs) -> Dict[str,float]:
    """
    Find the mixing coefficient between sets of observations in a causal graph using entropy/uncertainty-based confidence intervals.
    This function inspects the causal graph and determines the mixing coefficient between the expected outputs from two sets of observations. Only the outputs from the observed worlds are considered.
    The mixing coefficient is defined as the mean from the lower and upper bounds of the uncertainty interval of the paths from observations to the target node.
    The coefficient is calculated recursively, and the traversal can be limited by a specified cutoff depth.
    A higher mixing coefficient indicates a stronger influence of the set of observations 1 over the set of observations 2.
    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing causal relationships among nodes.
        target_node (str): The node for which the mixing coefficient is being calculated.
        observations_1 (Dict[str, Message]): A dictionary mapping node names to their corresponding messages representing the first set of observations.
        observations_2  (Dict[str, Message]): A dictionary mapping node names to their corresponding messages representing the second set of observations.
        traversal_cutoff (Optional[int]): Optional maximum depth for searching causal paths; if provided, limits the traversal.
        epsilon (float): A small positive value used to control the sensitivity of the mixing coefficient calculation. It should be between 0 and 1 (exclusive).
        world_set (Optional[List[nx.DiGraph]]): A list of causal graphs representing different instantiations of the causal network.
        factual_target_value (Optional[str]): The factual target value for the target node. If not provided, it is deduced from the observations.
        counterfactual_target_value (Optional[str]): The counterfactual target value for the target node. If not provided, it is deduced from the observations.
    Returns:
        float: The mixing coefficient between the two sets of observations with respect to the target node.
    """
    if epsilon <= 0.0 or epsilon >= 1.0:
        raise ValueError('Epsilon must be between 0 and 1')

    if world_set is None:
        world_set = [causal_graph]
    else:
        for world in world_set:
            if list(causal_graph.nodes) != list(world.nodes) or list(causal_graph.edges) != list(world.edges):
                raise ValueError(f'Worlds must have the same structure as the causal graph. Found {world.nodes} and {world.edges} instead of {causal_graph.nodes} and {causal_graph.edges}')


    _, entropies = observation_constrained_causal_graph_entropy(world_set, {**observations_1, **observations_2}, return_individual_entropies=True)
    for node in set(observations_1.keys()) | set(observations_2.keys()):
        entropies[node] = 0.0 # set the entropy of the observations to 0
    
    standard_deviations = {}
    for node, entropy in entropies.items():
        deviation = math.sqrt(math.exp(2 * entropy) / (2 * math.pi * math.e)) # standard deviation derived from the differential entropy (assuming a normal distribution)
        standard_deviations[node] = deviation
    confidence_interval_coeff = 1.96 # 95% confidence interval
    
    coeffs = {}
    def _find_mix_rec(node: str, depth: int) -> Tuple[float,float]: # lower and upper uncertainty bounds for coeff 
        if node in coeffs:
            return coeffs[node]
        if node in observations_1:
            return 1.0 - confidence_interval_coeff * standard_deviations[node], 1.0 + confidence_interval_coeff * standard_deviations[node]
        if node in observations_2:
            return -1.0 - confidence_interval_coeff * standard_deviations[node], -1.0 + confidence_interval_coeff * standard_deviations[node]
        if traversal_cutoff is not None and depth > traversal_cutoff:
            return 0.0, 0.0
        
        min_coeff = 0.0
        max_coeff = 0.0
        
        parents = list(causal_graph.predecessors(node))
        for parent in parents:
            parent_min, parent_max = _find_mix_rec(parent, depth + 1)
            min_coeff = min(parent_min, min_coeff)
            max_coeff = max(parent_max, max_coeff)
            
        if len(parents) > 0:
            min_coeff /= len(parents) # normalize the coefficient by the number of parents
            max_coeff /= len(parents) # normalize the coefficient by the number of parents
            
        min_coeff -= confidence_interval_coeff * standard_deviations[node]
        max_coeff += confidence_interval_coeff * standard_deviations[node]

        coeffs[node] = min_coeff, max_coeff
        return min_coeff, max_coeff
    
    min_coeff, max_coeff = _find_mix_rec(target_node, 0)
    coeff = (min_coeff + max_coeff) / 2 # average of the lower and upper bounds (unbiased estimate of the coefficient assuming symmetric uncertainty)
    coeff = 1 / (1 + math.exp(-coeff * (1 / epsilon))) # sigmoid function to normalize the coefficient between 0 and 1

    if not factual_target_value: # Observing the target node may lead to inconsistencies in the ground truth. TODO: assess if it should be allowed
        factual_target_value = observations_1[target_node]['current_value']
    if not counterfactual_target_value:
        counterfactual_target_value = observations_2[target_node]['current_value']

    ground_truth = {
        factual_target_value: coeff,
        counterfactual_target_value: 1 - coeff
    }
    return ground_truth



def find_mixing_coefficients_overlapping_beams(
        causal_graph: nx.DiGraph, 
        target_node: str, 
        observations_1: Dict[str, Message], 
        observations_2: Dict[str, Message],
        temperature: float = 1.0,
        traversal_cutoff: Optional[int] = None, # TODO: include in the entropy calculation
        world_set: Optional[List[nx.DiGraph]] = None, 
        **kwargs) -> Dict[str,float]:
    """
    Find the mixing coefficients between sets of observations in a causal graph using overlapping beams.
    This function inspects the causal graph and determines the mixing coefficients for each likely output given the set of observations.
    The mixing coefficients 
    The coefficient is calculated recursively, and the traversal can be limited by a specified cutoff depth.
    A higher mixing coefficient indicates a stronger influence of the set of observations 1 over the set of observations 2.
    Parameters:
        causal_graph (nx.DiGraph): A directed graph representing causal relationships among nodes.
        target_node (str): The node for which the mixing coefficient is being calculated.
        observations_1 (Dict[str, Message]): A dictionary mapping node names to their corresponding messages representing the first set of observations.
        observations_2  (Dict[str, Message]): A dictionary mapping node names to their corresponding messages representing the second set of observations.
        temperature (float): A small positive value used to control the sensitivity of the mixing coefficient calculation. It should be strictly positive.
        traversal_cutoff (Optional[int]): Optional maximum depth for searching causal paths; if provided, limits the traversal.
        world_set (Optional[List[nx.DiGraph]]): A list of causal graphs representing different instantiations of the causal network.
    Returns:
        float: The mixing coefficient between the two sets of observations with respect to the target node.
    """
    if temperature <= 0.0:
        raise ValueError('Temperature must be strictly positive')

    if world_set is None:
        world_set = [causal_graph]
    else:
        for world in world_set:
            if list(causal_graph.nodes) != list(world.nodes) or list(causal_graph.edges) != list(world.edges):
                raise ValueError(f'Worlds must have the same structure as the causal graph. Found {world.nodes} and {world.edges} instead of {causal_graph.nodes} and {causal_graph.edges}')
            
    _, beams = observation_constrained_causal_graph_entropy(world_set, {**observations_1, **observations_2}, return_individual_entropies=False, return_node_instances=True)

    # Etract the target node values from the set of possible worlds and the subset of worlds that are in the beam
    target_all_values = set(world.nodes[target_node]['current_value'] for world in world_set)
    target_beam_values = [node_instance['current_value'] for node_instance in beams[target_node]]

    # Calculate the mixing coefficients for each value in the target node. Add 1 to the count to improve the stability of the distribution
    value_counts = {value: 1 + target_beam_values.count(value) for value in target_all_values}
    z_t = sum([c**(1 / temperature) for c in value_counts.values()])
    target_value_probas = {value: (c**(1 / temperature)) / z_t for value, c in value_counts.items()}

    return target_value_probas




class BaseWorldManager(_WorldManager):

    def __init__(self, initial_graph: Optional[nx.DiGraph] = None, graphs: Optional[List[nx.DiGraph]] = None, traversal_cutoff: Optional[int] = None):
        super().__init__(initial_graph, graphs, traversal_cutoff)
        self.world_key_syntax = r'world_\d+'
        self.world_nodes = extract_world_nodes(self.graph, self.world_key_syntax)
        self.world_count = max([int(key.split('_')[-1]) for key in self.world_nodes.keys()]) + 1
    

    def merge(self, new_graph: nx.DiGraph, current_world: Optional[str] = None, **kwargs) -> None:
        if not current_world:
            current_world = f'world_{self.world_count}'
            self.world_count += 1
            self.world_nodes[current_world] = set()

        new_graph = new_graph.copy()
        grounded_attributes = list([VariableDefinition.__annotations__.keys(), _CausalEffectDefinition.__annotations__.keys()])
        for node, attrs in new_graph.nodes(data=True):
            self.world_nodes[current_world].add(node)

            world_attrs = {}
            for key in attrs.keys():
                if key in grounded_attributes:
                    world_attrs[key] = attrs[key]
                    del attrs[key]
            
            if node not in self.graph.nodes:
                self.graph.add_node(node, **attrs)
            else:
                self.graph.nodes[node].update(attrs) # TODO: current method retains the most recent attributes, assess if this is the desired behaviour
            
            self.graph.nodes[node][current_world] = world_attrs

        for source, target, attrs in new_graph.edges(data=True):
            if (source, target) not in self.graph.edges:
                self.graph.add_edge(source, target, **attrs)
            else:
                self.graph.edges[source, target].update(attrs)
    

    def get_world(self, world_id: str) -> nx.DiGraph:
        if world_id not in self.world_nodes:
            raise ValueError(f'World {world_id} not found')
        
        world_graph = self.graph.subgraph(self.world_nodes[world_id]).copy()
        world_graph = ground_in_world(world_graph, world_id, self.world_key_syntax)

        return world_graph
    
    
    def get_worlds(self) -> Dict[str, nx.DiGraph]:
        worlds = {}
        for world_id in self.world_nodes:
            worlds[world_id] = self.get_world(world_id)

        return worlds
    

    def get_worlds_from_node(self, target_node: str) -> Dict[str, nx.DiGraph]:
        worlds = {}
        for world_id, nodes in self.world_nodes.items():
            if target_node in nodes:
                world_graph = self.get_world(world_id)
                world_graph = nx.node_connected_component(world_graph, target_node) # Remove nodes not connected to target node

                worlds[world_id] = world_graph

        return worlds
    

    def generate_observations(self, target_node: str) -> Generator[Message, None, None]:
        for world_id, nodes in self.world_nodes.items():
            world_graph = self.get_world(world_id)
            if target_node in world_graph.nodes and 'current_value' in world_graph.nodes[target_node]:
                ground_truth = world_graph.nodes[target_node]['current_value']
                observations = {node: world_graph.nodes[node] for node in world_graph.nodes if node != target_node}

                exist_blanket = True
                while exist_blanket and len(observations) > 0:
                    try:
                        causal_blanket = build_target_node_causal_blanket(world_graph, target_node, observations.keys())
                        dags = dagify(causal_blanket)
                        blanket_observations = {node: world_graph.nodes[node] for node in causal_blanket if node != target_node}
                        observations = {node: world_graph.nodes[node] for node in observations if node not in blanket_observations}

                        for dag in dags:
                            yield Query(
                                world_ids=[world_id],
                                causal_graph=dag,
                                target_node=target_node,
                                ground_truth=ground_truth,
                                is_pseudo_gt=False,
                                observations=blanket_observations,
                            )
                    except nx.NetworkXNoPath:
                        exist_blanket = False
                    

    def _generate_counterfactuals(self, target_node: str, generator_func: Callable, num_interventions: int = 1) -> Generator[Query, None, None]:
        if num_interventions < 1:
            raise ValueError('Number of interventions must be greater than 0')

        # Extract candidate worlds
        candidate_worlds = self.get_worlds_from_node(target_node)

        # Establish causal blanket and dagify graph for inference
        worlds_with_blanket = {}
        for world_id, world_graph in candidate_worlds.items():
            if target_node in world_graph.nodes and 'current_value' in world_graph.nodes[target_node]:
                try:
                    causal_blanket = build_target_node_causal_blanket(world_graph, target_node, list(world_graph.nodes))
                    dag = dagify(causal_blanket)
                    worlds_with_blanket[world_id] = dag
                except nx.NetworkXNoPath:
                    continue
        
        for i, (world_id_1, world_graph_1) in enumerate(worlds_with_blanket.items()):
            for world_id_2, world_graph_2 in worlds_with_blanket.items()[i+1:]:
                observations_1_nodes = self.world_nodes[world_id_1]
                observations_2_nodes = self.world_nodes[world_id_2]

                if len(observations_1_nodes & observations_2_nodes) == 0: # No common observed nodes between worlds, counterfactuals cannot be matched
                    continue

                else:
                    common_dag = nx.DiGraph()
                    common_dag.add_nodes_from(set(world_graph_1.nodes) | set(world_graph_2.nodes))
                    common_dag.add_edges_from(set(world_graph_1.edges) | set(world_graph_2.edges))
                    common_dag = ground_in_world(common_dag, None, self.world_key_syntax)

                    shared_observations = find_shared_observations(
                        {node_name: world_graph_1.nodes[node_name] for node_name in observations_1_nodes}, 
                        {node_name: world_graph_2.nodes[node_name] for node_name in observations_2_nodes}
                        )
                    if len(shared_observations) == 0: # No shared observations between worlds, counterfactuals cannot be matched
                        continue

                    else:
                        for (factual_world_id, factual_graph), (counterfactual_world_id, counterfactual_graph) in itertools.permutations([(world_id_1, world_graph_1), (world_id_2, world_graph_2)], 2):
                            yield from generator_func(
                                factual_world_id=factual_world_id,
                                counterfactual_world_id=counterfactual_world_id,
                                common_dag=common_dag,
                                target_node=target_node,
                                factual_graph=factual_graph,
                                counterfactual_graph=counterfactual_graph,
                                shared_observations=shared_observations,
                                num_interventions=num_interventions,
                                world_set=worlds_with_blanket,
                            )

    def generate_counterfactuals_match(self, target_node: str, num_interventions: int = 1) -> Generator[Query, None, None]:
        def generator_func(factual_world_id, counterfactual_world_id, common_dag, target_node, factual_graph, counterfactual_graph, shared_observations, num_interventions, world_set):
            factual_observations = {
                node_name: factual_graph.nodes[node_name] for node_name in self.world_nodes[factual_world_id] if node_name not in shared_observations
            }
            counterfactual_observations = {
                node_name: counterfactual_graph.nodes[node_name] for node_name in self.world_nodes[counterfactual_world_id]
            }
            active_interventions = find_active_interventions(common_dag, target_node, factual_observations, counterfactual_observations, traversal_cutoff=self.traversal_cutoff)
            ground_truth = counterfactual_graph.nodes[target_node]['current_value']

            if len(active_interventions) < num_interventions: # Not enough active interventions to match counterfactuals
                return
            
            else:
                for intervention_combination in itertools.combinations(active_interventions, num_interventions):
                    selected_observations = find_non_blocking_observations(common_dag, target_node, factual_observations, shared_observations, intervention_combination, traversal_cutoff=self.traversal_cutoff)

                    yield Query(
                        world_ids=[factual_world_id, counterfactual_world_id],
                        causal_graph=common_dag,
                        target_node=target_node,
                        ground_truth=ground_truth,
                        is_pseudo_gt=False,
                        observations=selected_observations,
                        interventions=intervention_combination,
                    )

        return self._generate_counterfactuals(target_node, generator_func, num_interventions)


    def generate_counterfactuals_mix(self, target_node: str, num_interventions: int = 1, mixing_function: str ='linear') -> Generator[Query, None, None]:
        mix_funcs = {
            'linear': find_mixing_coefficient_linear,
            'intervals': find_mixing_coefficient_deviation_intervals,
            'overlapping_beams': find_mixing_coefficients_overlapping_beams,
        }

        def generator_func(factual_world_id, counterfactual_world_id, common_dag, target_node, factual_graph, counterfactual_graph, shared_observations, num_interventions, world_set):
            factual_observations = {
                node_name: factual_graph.nodes[node_name] for node_name in self.world_nodes[factual_world_id] if node_name not in shared_observations
            }
            counterfactual_observations = {
                node_name: counterfactual_graph.nodes[node_name] for node_name in self.world_nodes[counterfactual_world_id]
            }
            active_interventions = find_active_interventions(common_dag, target_node, factual_observations, counterfactual_observations, traversal_cutoff=self.traversal_cutoff)

            ground_truth = mix_funcs[mixing_function](
                common_dag, target_node, factual_observations, counterfactual_observations, 
                traversal_cutoff=self.traversal_cutoff, world_set=world_set, 
                factual_target_value=factual_graph.nodes[target_node]['current_value'], 
                counterfactual_target_value=counterfactual_graph.nodes[target_node]['current_value'])

            if len(active_interventions) < num_interventions: # Not enough active interventions to match counterfactuals
                return
            
            else:
                for intervention_combination in itertools.combinations(active_interventions, num_interventions):
                    yield Query(
                        world_ids=[factual_world_id, counterfactual_world_id],
                        causal_graph=common_dag,
                        target_node=target_node,
                        ground_truth=ground_truth,
                        is_pseudo_gt=True,
                        observations=factual_observations,
                        interventions=intervention_combination,
                    )

        return self._generate_counterfactuals(target_node, generator_func, num_interventions)



                
__all__ = ['BaseWorldManager'] 


