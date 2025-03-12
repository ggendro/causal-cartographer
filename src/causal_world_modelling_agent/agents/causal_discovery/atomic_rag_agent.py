
from typing import Optional
import networkx as nx

from smolagents import Model

from ...utils.graph_utils import is_digraph
from ..factory import AgentFactory
from ...syntax.messages import OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP
from ...tools.retrieval import GraphRetrieverTool
from ..custom_prompt_agent import CustomPromptAgent


class AtomicRAGDiscoveryAgent(CustomPromptAgent):

    def run(self, task: str, *args, **kwargs) -> nx.DiGraph:
        retrieval_tool = self.tools["graph_retriever"]
        rag_task = task + retrieval_tool(task)

        return super().run(rag_task, *args, **kwargs)


class AtomicRAGDiscoveryAgentFactory(AgentFactory[AtomicRAGDiscoveryAgent]):
    
    def __init__(self, path_to_system_prompt: str = 'rag_causal_discovery.yaml', use_prompt_lib_folder: bool = True, graph_save_path: Optional[str] = None, initial_graph: Optional[nx.DiGraph] = None, depth: int = 2, max_documents: int = 3):
        super().__init__(AtomicRAGDiscoveryAgent, path_to_system_prompt, use_prompt_lib_folder)

        assert not (graph_save_path and initial_graph), "Either provide a graph save path or an initial graph, not both."
        
        self.graph_save_path = graph_save_path
        self.initial_graph = initial_graph
        self.depth = depth
        self.max_documents = max_documents
        self.retrieval_tool = None

        if self.graph_save_path:
            self.initial_graph = nx.read_gml(self.graph_save_path)


    def createAgent(self, base_model: Model, *args, api_key: Optional[str] = None, **kwargs) -> AtomicRAGDiscoveryAgent:

        if not api_key:
            try:
                api_key = base_model.api_key
            except AttributeError:
                raise AttributeError("No API key provided and no API key found in the base model. Please provide an API key as a factory argument or in the base model.")

        self.retrieval_tool = GraphRetrieverTool(
            graph=None if not self.initial_graph else self.initial_graph.copy(), 
            depth=self.depth, 
            max_documents=self.max_documents,
            api_key=api_key
        )

        return super().createAgent(
            *args,
            tools=[self.retrieval_tool],
            base_model=base_model,
            additional_system_prompt_variables={
                'observed_variable': OBSERVED_VARIABLE,
                'variable': VARIABLE,
                'causal_relationship': CAUSAL_RELATIONSHIP,
                'retrieval_tool_name': self.retrieval_tool.name
            },
            final_answer_checks=[is_digraph],
            **kwargs
        )