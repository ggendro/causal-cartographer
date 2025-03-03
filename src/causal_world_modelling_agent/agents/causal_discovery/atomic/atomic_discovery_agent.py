
from smolagents import Model

from ....core.agent import CustomSystemPromptCodeAgent
from ...factory import AgentFactory
from ....syntax.messages import EVENT, OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP


class AtomicDiscoveryAgentFactory(AgentFactory):

    DESCRIPTION = """
                    Agent that extracts a networkx causal graph from a text snippet.
                    """

    SYSTEM_PROMPT =f"""You are an agent that extracts a networkx causal graph from a text snippet. The manager plan is as follows:
                    1. Retrieve the events in the text. Events have the following format:
                    {EVENT}
                    2. Retrieve the causal variables in the text associated with each event. Causal variables have the following format:
                    {OBSERVED_VARIABLE}
                    Some variables may be confounders that are not explicitely given in the text but affect the system. Estimate the confounders and add them to the list of causal variables. Their values are not directly available, therefore their expected format is as follows:
                    {VARIABLE}
                    3. Build a partial order from the list of causal variables. Use common sense knowledge to estimate the order of the causal variables and return a list of tuples, where each tuple corresponds to a pair of non-identical variable names that should be ordered.
                    4. Build the full causal graph as a networkx graph. The agent must be provided with the list of causal variables and will return a networkx DiGraph object. Causal relationships have the following format:
                    {CAUSAL_RELATIONSHIP}
                    5. The agent must return the networkx causal graph as a final answer.

                    Here is an example of task execution:
                    ---
                    Task: [INPUT_DOCUMENT]

                    Thought: I will proceed step by step and extract the causal graph from the text.
                    First, I will find the events in the text, then I will find the causal variables associated with the event and build a partial order from the list of causal variables. Finally, I will build the full causal graph.
                    Code:
                    ```py
                    # Step 1. Define the event from the text
                    event = {{
                        "name": "...",
                        ...
                    }}

                    # Step 2. Define the causal variables with details
                    causal_variables = [
                        {{
                            "name": "...",
                            ...
                        }},
                        ...
                    ]

                    # Step 3. Define partially ordered pairs of causal variables based on assumed causal relationships. Each tuple is in the form (cause, effect).                                                      
                    partial_order = [
                        ("...", "..."),
                        ...
                    ]

                    # Step 4. Build the full causal graph as a networkx DiGraph and add the causal variables as nodes.
                    G = nx.DiGraph()

                    # Add nodes with additional details representing the causal variables.
                    for var in causal_variables:
                        var_name = variable["name"]
                        G.add_node(var_name)
                        G.nodes[var_name].update(variable)

                    # Add edges with additional details representing the causal relationships.                                                                 
                    G.add_edge("...", "...", description="...", ...)
                    ...

                    # Step 5. Return the networkx causal graph as the final answer.
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

    def createAgent(self, base_model: Model) -> CustomSystemPromptCodeAgent:
        return CustomSystemPromptCodeAgent(
                    tools=[], 
                    model=base_model, 
                    additional_authorized_imports=["networkx"],
                    name="end_to_end_causal_extraction_agent", 
                    description=AtomicDiscoveryAgentFactory.DESCRIPTION,
                    custom_system_prompt=AtomicDiscoveryAgentFactory.SYSTEM_PROMPT,
        )