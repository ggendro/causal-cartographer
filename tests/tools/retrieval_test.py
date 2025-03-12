
import networkx as nx
import pytest
import re

from mocks.mock_embedding_models import MockEmbeddings
from causal_world_modelling_agent.tools.retrieval import GraphRetrieverTool


class TestGraphRetrievalTool:

    @pytest.fixture
    def graph_retriever(self):
        return GraphRetrieverTool(embeddings=MockEmbeddings())
    
    @pytest.fixture
    def graph(self):
        graph = nx.DiGraph()
        graph.add_node("A", description="Node A")
        graph.add_node("B", description="Node B")
        graph.add_edge("A", "B", description="Edge A -> B")
        return graph
    
    @pytest.fixture
    def graph_retriever_with_data(self, graph):
        return GraphRetrieverTool(graph=graph, embeddings=MockEmbeddings())
    
    @pytest.fixture
    def large_graph(self):
        graph = nx.DiGraph()
        for i in range(100):
            graph.add_node(str(i), description=f"Node {i}")
            if i > 0:
                graph.add_edge(str(i - 1), str(i), description=f"Edge {i - 1} -> {i}")
        return graph
    
    @pytest.fixture
    def graph_retriever_with_large_data_no_depth(self, large_graph):
        return GraphRetrieverTool(graph=large_graph, embeddings=MockEmbeddings(), depth=0, max_documents=12)
    
    @pytest.fixture
    def graph_retriever_with_large_data_one_document(self, large_graph):
        return GraphRetrieverTool(graph=large_graph, embeddings=MockEmbeddings(), depth=10, max_documents=1)
    
    
    def test_get_graph(self, graph_retriever):
        assert isinstance(graph_retriever.get_graph(), nx.DiGraph)
        assert len(graph_retriever.get_graph().nodes) == 0
        assert len(graph_retriever.get_graph().edges) == 0

    def test_get_graph_with_data(self, graph_retriever_with_data, graph):
        assert list(graph_retriever_with_data.get_graph().nodes(data=True)) == list(graph.nodes(data=True))
        assert list(graph_retriever_with_data.get_graph().edges(data=True)) == list(graph.edges(data=True))

    def test_update_graph(self, graph_retriever, graph):
        graph_retriever.update_graph(graph)
        assert list(graph_retriever.get_graph().nodes(data=True)) == list(graph.nodes(data=True))
        assert list(graph_retriever.get_graph().edges(data=True)) == list(graph.edges(data=True))

    def test_get_documents(self, graph_retriever_with_data):
        documents_A = graph_retriever_with_data.retriever.vectorstore.get_by_ids('A')
        assert len(documents_A) == 1
        assert documents_A[0].id == 'A'
        assert documents_A[0].page_content == "A\nNode A"
        
        documents_B = graph_retriever_with_data.retriever.vectorstore.get_by_ids('B')
        assert len(documents_B) == 1
        assert documents_B[0].id == 'B'
        assert documents_B[0].page_content == "B\nNode B"

    def test_forward(self, graph_retriever_with_data):
        node_reg = re.compile(r"(\w+): (Node .+)")
        edge_reg = re.compile(r"(\w+) -> (\w+): (.+)")

        retrieved_text = graph_retriever_with_data.forward("A")
        retrieved_nodes = node_reg.findall(retrieved_text)
        retrieved_edges = edge_reg.findall(retrieved_text)

        assert len(retrieved_nodes) == 2
        assert set(retrieved_nodes) == {("A", "Node A"), ("B", "Node B")}

        assert len(retrieved_edges) == 1
        assert retrieved_edges[0] == ("A", "B", "Edge A -> B")

    def test_forward_with_large_graph_no_depth(self, graph_retriever_with_large_data_no_depth):
        node_reg = re.compile(r"(\w+): (Node .+)")
        edge_reg = re.compile(r"(\w+) -> (\w+): (.+)")

        retrieved_text = graph_retriever_with_large_data_no_depth.forward("0")
        retrieved_nodes = node_reg.findall(retrieved_text)
        retrieved_edges = edge_reg.findall(retrieved_text)

        assert len(retrieved_nodes) == 12
        assert len(retrieved_edges) == 0

    def test_forward_with_large_graph_one_document(self, graph_retriever_with_large_data_one_document):
        node_reg = re.compile(r"(\w+): (Node .+)")
        edge_reg = re.compile(r"(\w+) -> (\w+): (.+)")

        retrieved_text = graph_retriever_with_large_data_one_document.forward("50")
        retrieved_nodes = node_reg.findall(retrieved_text)
        retrieved_edges = edge_reg.findall(retrieved_text)

        assert 10 <= len(retrieved_nodes) <= 21
        assert len(retrieved_nodes) == len(retrieved_edges) + 1

        answer_graph = nx.DiGraph()
        answer_graph.add_nodes_from([(name, {"description": description}) for name, description in retrieved_nodes])
        answer_graph.add_edges_from([(start, end, {"description": description}) for start, end, description in retrieved_edges])

        assert nx.is_directed_acyclic_graph(answer_graph)
        assert nx.is_weakly_connected(answer_graph)

        sources = [node for node in answer_graph.nodes if answer_graph.in_degree(node) == 0]
        sinks = [node for node in answer_graph.nodes if answer_graph.out_degree(node) == 0]
        
        assert len(sources) == 1
        assert len(sinks) == 1
        
        node = sources[0]
        while answer_graph.out_degree(node) > 0:
            assert answer_graph.out_degree(node) == 1
            node = list(answer_graph.successors(node))[0]

        answer_graph.add_edge(sinks[0], sources[0])
        assert nx.is_strongly_connected(answer_graph)