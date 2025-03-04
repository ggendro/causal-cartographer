
from typing import Optional
import networkx as nx

from smolagents import Model

from ...core.agent import CustomSystemPromptCodeAgent
from ..factory import AgentFactory
from ...syntax.messages import OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP
from ...tools.retrieval import GraphRetrieverTool


class AtomicRAGDiscoveryAgent(CustomSystemPromptCodeAgent):

    def run(self, task: str, *args, **kwargs) -> nx.DiGraph:
        retrieval_tool = self.tools["graph_retriever"]
        rag_task = task + retrieval_tool(task)

        print(rag_task)

        return super().run(rag_task, *args, **kwargs)


class AtomicRAGDiscoveryAgentFactory(AgentFactory):
    
    def __init__(self, path_to_prompt_syntax: Optional[str] = None, graph_save_path: Optional[str] = None, initial_graph: Optional[nx.DiGraph] = None, depth: int = 2, max_documents: int = 3, api_key: Optional[str] = None):
        if not path_to_prompt_syntax:
            import os
            path_to_prompt_syntax = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'syntax', 'prompts', 'rag_causal_discovery.yaml')
        
        super().__init__(path_to_prompt_syntax)

        assert not (graph_save_path and initial_graph), "Either provide a graph save path or an initial graph, not both."
        
        self.graph_save_path = graph_save_path
        self.initial_graph = initial_graph
        self.depth = depth
        self.max_documents = max_documents
        self.api_key = api_key
        self.retrieval_tool = None

        if self.graph_save_path:
            self.initial_graph = nx.read_gml(self.graph_save_path)


    def createAgent(self, base_model: Model) -> CustomSystemPromptCodeAgent:

        if self.api_key:
            api_key = self.api_key
        else:
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

        system_prompt = self.system_prompt.format(
            observed_variable=OBSERVED_VARIABLE,
            variable=VARIABLE,
            causal_relationship=CAUSAL_RELATIONSHIP,
            retrieval_tool_name=self.retrieval_tool.name
        )

        return AtomicRAGDiscoveryAgent(
            tools=[self.retrieval_tool],
            model=base_model, 
            additional_authorized_imports=["networkx"],
            name="rag_causal_extraction_agent", 
            description=self.description,
            custom_system_prompt=system_prompt,
        )