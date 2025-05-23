
from typing import Optional, List
import networkx as nx
from collections import deque

from smolagents import Model


from ...utils.graph_utils import isDigraph
from ...utils.message_utils import isGraphMessageDefinition
from ..factory import AgentFactory
from ...syntax.definitions import VariableDefinition, CausalRelationshipDefinition
from ..custom_prompt_agent import CustomPromptAgent



class SelfIterativeDiscoveryAgent(CustomPromptAgent):

    def __init__(self, *args, pre_prompt: str, num_iterations: int = 1, initial_graph: Optional[nx.DiGraph] = None, previous_history: Optional[List[nx.DiGraph]] = None, **kwargs):
        assert num_iterations > 0, "The number of iterations must be greater than 0."
        self.num_iterations = num_iterations
        self.pre_prompt = pre_prompt

        if initial_graph is not None:
            self.causal_graph = initial_graph
        else:
            self.causal_graph = nx.DiGraph()

        if previous_history is not None:
            self.history = previous_history
        else:
            self.history = []

        super().__init__(*args, **kwargs)

    def run(self, task: str, *args, **kwargs) -> nx.DiGraph:
        queue = deque([task])

        for _ in range(self.num_iterations):
            if not queue:
                break

            formatted_task = self.pre_prompt.format(topic=queue.popleft(), nodes=dict(self.causal_graph.nodes(data=True)), edges=list(self.causal_graph.edges(data=True)))
            updated_graph = super().run(formatted_task, *args, additional_args={'G': self.causal_graph.copy()}, **kwargs)

            added_nodes = set(updated_graph.nodes) - set(self.causal_graph.nodes)
            for new_node in added_nodes:
                queue.append(new_node)

            self.history.append(updated_graph.copy())
            self.causal_graph = updated_graph
        return self.causal_graph



class SelfIterativeDiscoveryAgentFactory(AgentFactory[SelfIterativeDiscoveryAgent]):

    def __init__(self, path_to_system_prompt: str = 'self_iteration_discovery_truncated.yaml', use_prompt_lib_folder: bool = True, num_iterations: int = 1, graph_save_path: Optional[str] = None, initial_graph: Optional[nx.DiGraph] = None, previous_history: Optional[List[nx.DiGraph | str]] = None):
        super().__init__(SelfIterativeDiscoveryAgent, path_to_system_prompt, use_prompt_lib_folder)
        
        self.num_iterations = num_iterations

        assert not (graph_save_path and initial_graph), "Either provide a graph save path or an initial graph, not both."
        
        self.graph_save_path = graph_save_path
        self.initial_graph = initial_graph
        self.previous_history = previous_history

        if self.graph_save_path:
            self.initial_graph = nx.read_gml(self.graph_save_path)
        
        if self.previous_history:
            self.previous_history = [nx.read_gml(graph) if isinstance(graph, str) else graph for graph in self.previous_history]

        if self.previous_history and not self.initial_graph:
            self.initial_graph = self.previous_history[-1]


    def createAgent(self, base_model: Model, *args, **kwargs) -> SelfIterativeDiscoveryAgent:
        return super().createAgent(
            *args,
            base_model=base_model,
            pre_prompt=self.user_pre_prompt,
            num_iterations=self.num_iterations,
            initial_graph=self.initial_graph,
            previous_history=self.previous_history,
            additional_system_prompt_variables={
                'variable': VariableDefinition.get_definition(),
                'causal_relationship': CausalRelationshipDefinition.get_definition()
            },
            final_answer_checks=[isDigraph, isGraphMessageDefinition],
            **kwargs
        )