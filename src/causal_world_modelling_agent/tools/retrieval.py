
import networkx as nx
from collections import deque
from typing import Optional

from smolagents import Tool
from langchain_core.vectorstores import InMemoryVectorStore # TODO: switch to more advanced vector databases for handling larger graphs, see https://python.langchain.com/docs/concepts/vectorstores/
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings


class GraphRetrieverTool(Tool): # from https://huggingface.co/docs/smolagents/examples/rag and https://python.langchain.com/docs/integrations/retrievers/graph_rag/
    name = "graph_retriever"
    description = "Uses semantic search to retrieve the subpgraph of the causal knowledge graph database that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target nodes. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, graph: Optional[nx.DiGraph] = None, depth: int = 2, max_documents: int = 3, api_key: Optional[str] = None, embeddings: Optional[Embeddings] = None, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth
        self.max_documents = max_documents

        if not embeddings:
            embeddings = OpenAIEmbeddings(api_key=api_key)

        self.retriever = InMemoryVectorStore(embeddings).as_retriever()
        self.graph = nx.DiGraph()

        if graph:
            self.update_graph(graph)

    def update_graph(self, graph: nx.DiGraph):
        documents = {}
        for node_name, attrs in graph.nodes(data=True):
            documents[node_name] = Document(
                id=node_name,
                page_content=f"{node_name}\n{attrs.get('description', '')}",
                metadata=attrs
            )
        self.retriever.add_documents(documents.values(), ids=list(documents.keys()))

        self.graph.add_nodes_from(graph.nodes(data=True))
        self.graph.add_edges_from(graph.edges(data=True))

    def get_graph(self):
        return self.graph
        

    def forward(self, query: str) -> str:
        docs = self.retriever.invoke(query, k=self.max_documents)

        nodes, edges = {}, {}
        queue = deque([(doc.id, self.depth) for doc in docs])

        while queue:
            node, depth = queue.popleft()

            nodes[node] = self.graph.nodes[node]

            if depth > 0:
                for neighbor in self.graph.neighbors(node):
                    edges[(node, neighbor)] = self.graph.edges[(node, neighbor)]

                    if neighbor not in nodes:
                        queue.append((neighbor, depth - 1))

                for neighbor in self.graph.predecessors(node):
                    edges[(neighbor, node)] = self.graph.edges[(neighbor, node)]

                    if neighbor not in nodes:
                        queue.append((neighbor, depth - 1))


        return "\nRetrieved nodes:\n" + "".join(
            [
                f"{node}: {attrs.get('description', '')}\n"
                for node, attrs in nodes.items()
            ] + [
                f"\nRetrieved edges:\n" + "".join(
                    [
                        f"{edge[0]} -> {edge[1]}: {attrs.get('description', '')}\n"
                        for edge, attrs in edges.items()
                    ]
                )
            ]
        )