from typing import Optional
import networkx as nx

from smolagents import Model

from ....core.agent import CustomSystemPromptCodeAgent
from ...factory import AgentFactory
from ....syntax.messages import OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP
from ....tools.retrieval import GraphRetrieverTool


class AtomicRAGDiscoveryAgent(CustomSystemPromptCodeAgent):

    def run(self, task: str, *args, **kwargs) -> nx.DiGraph:
        retrieval_tool = self.tools["graph_retriever"]
        rag_task = task + retrieval_tool(task)

        print(rag_task)

        return super().run(rag_task, *args, **kwargs)


class AtomicRAGDiscoveryAgentFactory(AgentFactory):

    DESCRIPTION = """
                    Agent that extracts a networkx causal graph from a text snippet.
                    """

    SYSTEM_PROMPT ="""You are an agent that extracts a networkx causal graph from a text snippet. An initial causal graph is provided and can be queried to verify if a variable already exists. The manager plan is as follows:
                    1. Retrieve the causal variables in the text associated with each event. An initila subset of variables is provided as a help. Use the variables provided when possible or create new ones when no variable matches. Causal variables have the following format:
                    {observed_variable}
                    Some variables may be confounders that are not explicitely given in the text but affect the system. Estimate the confounders and add them to the list of causal variables. Their values are not directly available, therefore their expected format is as follows:
                    {variable}
                    2. Verify if the newly variables have correspondance in the causal graph database. Use the `{retrieval_tool_name}` tool to assess if the variable is already in the database. If it is, use it instead of creating a new one. It may have a different name in the database, the tool returns the top-k matching variables. use the one matching the most or create a new one if none matches.
                    3. Build the full causal graph as a networkx graph. The agent must be provided with the list of causal variables and will return a networkx DiGraph object. Do not create causal relationships that already exist in the causal graph. Causal relationships have the following format:
                    {causal_relationship}
                    4. The agent must return the networkx causal graph as a final answer.

                    Here is an example of task execution. The code MUST be executed in two code blocks. After step 2, use <end_code> to indicate the end of the code block and retrieve the output of the tool call. Then, use the observation to execute steps 3 and 4 and complete the task.
                    ---
                    Task: [INPUT_DOCUMENT]

                    Thought: I will proceed step by step and extract the causal graph from the text.
                    First, I will find the causal variables and verify if they exist. Then, I will build the full causal graph.
                    Code:
                    ```py
                    # Step 1. Define the causal variables with details
                    causal_variables = [
                        {{
                            "name": "...",
                            ...
                        }},
                        ...
                    ]
                    
                    # Step 2. Verify if the variables exist in the causal graph database
                    for var in causal_variables:
                        retrieval_tool_answer = {retrieval_tool_name}(task="Is the following variable in the database? var['name']: var['description']")
                        print(retrieval_tool_answer)
                    ```<end_code>
                    Observation: Retreieved nodes: ... Retrieverd edges: ...
                    
                    Code:
                    ```py
                    # Step 3. Build the full causal graph as a networkx DiGraph and add the causal variables as nodes.
                    G = nx.DiGraph()

                    # Update node list and add nodes with additional details representing the causal variables. Only add the variables that do not exist in the causal graph.
                    updated_causal_variables = [
                        {{
                            "name": "...",
                            ...
                        }},
                        ...
                    ]
                    
                    for var in updated_causal_variables:
                        var_name = variable["name"]
                        G.add_node(var_name)
                        G.nodes[var_name].update(variable)

                    # create edge list and add additional details representing the causal relationships. Only add the relationships that do not already exist in the causal graph.
                    updated_causal_relationships = [
                        {{
                            "cause" : "...",
                            "effect" : "...",
                            ...
                        }},
                        ...
                    ]

                    G.add_edge("...", "...", description="...", ...) # only adding 
                    ...

                    # Step 4. Return the networkx causal graph as the final answer.
                    print("Event extracted:")
                    print(event)
                    print("\\nCausal Variables:")
                    for var in causal_variables:
                        print(var)
                    print("\\nPartial Ordering (Cause -> Effect):")
                    for order in partial_order:
                        print(order)
                    print("\\nCausal Graph Nodes and Edges:")
                    print("Nodes:")
                    print(G.nodes(data=True))
                    print("Edges:")
                    print(list(G.edges(data=True)))

                    final_answer(G)
                    ```<end_code>
                    """

    
    def __init__(self, graph_save_path: Optional[str] = None, initial_graph: Optional[nx.DiGraph] = None, depth: int = 2, max_documents: int = 3, api_key: Optional[str] = None):
        
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

        system_prompt = AtomicRAGDiscoveryAgentFactory.SYSTEM_PROMPT.format(
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
                    description=AtomicRAGDiscoveryAgentFactory.DESCRIPTION,
                    custom_system_prompt=system_prompt,
        )