
from typing import Optional, List
import networkx as nx
from collections import deque

from smolagents import Model

from ....core.definitions import Message
from ....core.agent import CustomSystemPromptCodeAgent
from ...factory import AgentFactory
from ....syntax.messages import OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP



class SelfIterativeDiscoveryAgent(CustomSystemPromptCodeAgent):

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
            print(f"Iteration {_ + 1} / {self.num_iterations}")
            if not queue:
                break

            formatted_task = self.pre_prompt.format(topic=queue.popleft())
            updated_graph = super().run(formatted_task, *args, additional_args={'G': self.causal_graph.copy()}, **kwargs)

            added_nodes = set(updated_graph.nodes) - set(self.causal_graph.nodes)
            for new_node in added_nodes:
                queue.append(new_node)

            self.history.append(updated_graph.copy())
            self.causal_graph = updated_graph
        return self.causal_graph



class SelfIterativeDiscoveryAgentFactory(AgentFactory):

    def __init__(self, num_iterations: int = 1, graph_save_path: Optional[str] = None, initial_graph: Optional[nx.DiGraph] = None, previous_history: Optional[List[nx.DiGraph | str]] = None):
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


    DESCRIPTION = """
                    Agent that extracts a networkx causal graph on a given topic or from a text snippet, by iteratively adding elements to the graph.
                    """
    
    USER_PRE_PROMPT = """Build and expand a causal graph on the following subject:\n{topic}"""

    SYSTEM_PROMPT =f"""You are an agent that iteratively builds a networkx causal graph on a given topic or from  a text snippet. You take a causal graph `G` in input and adds element to it. The plan is as follows:
                    1. Print the variables and relationships that exist in the causal graph.
                    2. Provide additional causal variables and causal relationships to expand the graph. For each addition, at least one node must be connected to the existing graph.
                    3. The agent must return the updated networkx causal graph as a final answer.
                    Here is the format of  the variables:
                    {VARIABLE}
                    If an evidence text is provided in input, `current_value` and `contextual_information` should be filled with the corresponding values from the text, as follows:
                    {OBSERVED_VARIABLE}
                    Causal relationships have the following format:
                    {CAUSAL_RELATIONSHIP}
                    The causal graph is given as a networkx DiGraph object available as a parameter `G`.

                    Here is an example of task execution. Follow RIGOROUSLY the given format. First, read the input graph in ONE code block and END it with <end_code>. THEN, add a SECOND code block adding the variables and relationships, and finally end with the final answer.
                    ---
                    Task: "{USER_PRE_PROMPT.format(topic='impact of exercise on mental health')}"

                    Thought: First, let's extract the existing causal graph on the impact of exercise on mental health and print the variables and relationships that exist in the graph.
                    Code:
                    ```py
                    import networkx as nx

                    # Print the variables and relationships that exist in the graph.
                    print("Variables in the graph:", G.nodes(data=True))
                    print("Relationships in the graph:", G.edges(data=True))
                    ```<end_code>
                    Observation:
                    Variables in the graph: NodeDataView({{'Exercise Frequency': {{'name': 'Exercise Frequency', 'description': 'The number of exercise sessions per week', 'type': 'integer', 'values': [0, 1, 2, 3, 4, 5, 6, 7]}}, 'Physical Fitness': {{'name': 'Physical Fitness', 'description': "A measure of an individual's physical condition", 'type': 'string', 'values': ['poor', 'average', 'good', 'excellent']}}, 'Stress Levels': {{'name: 'Stress Levels', 'description': 'The perceived stress levels experienced by an individual', 'type': 'string', 'values': ['low', 'medium', 'high']}}, 'Mental Health': {{'name: 'Mental Health', 'description': "A measure of an individual's psychological and emotional well-being", 'type': 'string', 'values': ['poor', 'average', 'good', 'excellent']}}}}
                                  Relationships in the graph: OutEdgeDataView([('Exercise Frequency', 'Physical Fitness', {{'description': 'More frequent exercise sessions improve physical fitness.', 'contextual information': 'Regular exercise enhances muscle strength and cardiovascular health.', 'type': 'direct', 'strength': 'moderate', 'confidence': 'high', 'function': None}}), ('Exercise Frequency', 'Stress Levels', {{'description': 'Increased exercise frequency helps reduce stress levels.', 'contextual information': 'Regular physical activity can lower cortisol levels and alleviate stress.', 'type': 'direct', 'strength': 'moderate', 'confidence': 'high', 'function': None}})])
                    
                    Thought: The concept of 'Cognitive Function' and 'Sleep Quality' are missing in the current graph. Additionally, the relationship between 'Stress Levels' and 'Mental Health' is not present. Let's now add these variables and relationships to the graph.
                    Code:
                    ```py
                    # Provide additional causal variables and causal relationships to expand the graph.
                    # Define the causal variables with details
                    causal_variables = [
                        {{'name': 'Cognitive Function', 'description': 'The cognitive abilities of an individual', 'type': 'string', 'values': ['poor', 'average', 'good', 'excellent'], 'current_value': None, 'contextual_information': None}},
                        {{'name': 'Sleep Quality', 'description': 'The quality of sleep experienced by an individual', 'type': 'string', 'values': ['poor', 'average', 'good', 'excellent'], 'current_value': None, 'contextual_information': None}}
                    ]

                    # Define the causal relationships with details
                    causal_relationships = [
                        {{'cause': 'Stress Levels', 'effect': 'Mental Health', 'description': 'High stress levels negatively impact mental health.', 'contextual_information': 'Chronic stress can lead to anxiety and depression.', 'type': 'direct', 'strength': 'strong', 'confidence': 'high', 'function': None}}
                        {{'cause': 'Exercise Frequency', 'effect': 'Cognitive Function', 'description': 'Regular exercise improves cognitive function.', 'contextual_information': 'Physical activity enhances cognitive abilities and memory.', 'type': 'direct', 'strength': 'moderate', 'confidence': 'high', 'function': None}},
                        {{'cause': 'Sleep Quality', 'effect': 'Mental Health', 'description': 'Good sleep quality positively impacts mental health.', 'contextual_information': 'Adequate sleep is essential for emotional well-being and mental clarity.', 'type': 'direct', 'strength': 'moderate', 'confidence': 'high', 'function': None}}
                    ]

                    # Add the causal variables and relationships to the graph.
                    for variable in causal_variables:
                        var_name = variable["name"]
                        G.add_node(var_name)
                        G.nodes[var_name].update(variable)

                    for relationship in causal_relationships:
                        cause = relationship["cause"]
                        effect = relationship["effect"]
                        G.add_edge(cause, effect)
                        G[cause][effect].update(relationship)

                    final_answer(G)
                    ```<end_code>
                    """

    def createAgent(self, base_model: Model) -> SelfIterativeDiscoveryAgent:
        return SelfIterativeDiscoveryAgent(
            pre_prompt=self.USER_PRE_PROMPT,
                    tools=[], 
                    model=base_model, 
                    additional_authorized_imports=["networkx"],
                    name="self_iterative_discovery_agent", 
                    description=SelfIterativeDiscoveryAgentFactory.DESCRIPTION,
                    custom_system_prompt=SelfIterativeDiscoveryAgentFactory.SYSTEM_PROMPT,
                    num_iterations=self.num_iterations,
                    initial_graph=self.initial_graph,
                    previous_history=self.previous_history
        )