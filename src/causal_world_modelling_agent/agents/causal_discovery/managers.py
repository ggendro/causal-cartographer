
from smolagents import Model

from ...core.agent import CustomSystemPromptCodeAgent
from ..factory import AgentFactory
from .components.retrieval_agent import RetrievalAgentFactory
from .components.causal_order_agent import CausalOrderAgentFactory
from .components.causal_discovery_agent import CausalDiscoveryAgentFactory
from ...syntax.messages import EVENT, OBSERVED_VARIABLE, VARIABLE, CAUSAL_RELATIONSHIP


class CausalDiscoveryManagerAgentFactory(AgentFactory):

    SYSTEM_PROMPT ="""You are an agent that manages other agents in the process of extracting a causal graph from a text snippet. The manager plan is as follows:
                    1. Retrieve the events in the text. Events have the following format:
                    {event}
                    2. Interact with the {retrieval_agent_name} agent to assess if the event is already in the database. If not, add it. If it is, retrieve and complete its information. 
                    Events may have been stored in the database with a different name. The {retrieval_agent_name} agent will find the closest match, which must be used.
                    3. Retrieve the causal variables in the text associated with each event. Harness the variable names retrieved from the events in the database to complete the list. Causal variables have the following format:
                    {observed_variable}
                    Some variables may be confounders that are not explicitely given in the text but affect the system. Estimate the confounders and add them to the list of causal variables. Their values are not directly available, therefore their expected format is as follows:
                    {variable}
                    4. Interact with the {retrieval_agent_name} agent to assess if the variable is already in the database. If it is, retrieve and complete its information. Particularly, use the name already existing in the database.
                    Variables may have been stored in the database with a different name. The {retrieval_agent_name} agent will find the closest match, which must be used.
                    5. Interact with the {causal_order_agent_name} agent to build a partial order from the list of causal variables. 
                    The agent will use common sense knowledge to estimate the order of the causal variables and return a list of tuples, where each tuple corresponds to a pair of non-identical variable names that should be ordered.
                    6. Interact with the {causal_discovery_agent_name} agent to build the full causal graph. The agent must be provided with the list of causal variables and will return a networkx DiGraph object. Causal relationships have the following format:
                    {causalrelationship}
                    7. Interact with the {retrieval_agent_name} agent to add all the causal relationships to the database.
                    8. The agent must return the causal graph as a final answer.
                    Do not use external libraries. Let the other agents take care of the details.

                    Here is an example of task execution:
                    ---
                    Task: [INPUT_DOCUMENT]

                    Thought: I will proceed step by step and use the {retrieval_agent_name}, {causal_order_agent_name}, and {causal_discovery_agent_name} agents to extract the causal graph from the text.
                    First, I will find the events in the text and interact with the {retrieval_agent_name} agent to assess if the event is already in the database and if so, retrieve and complete its information.
                    The text contains one event with name [EVENT_NAME] and properties [EVENT_PROPERTIES]. Let's verify if it is in the database.
                    Code:
                    ```py
                    retrieval_agent_answer = {retrieval_agent_name}(task="Is the following event in the database? [EVENT_NAME]: [EVENT_PROPERTIES]")
                    print(retrieval_agent_answer)
                    ```<end_code>
                    Observation: The event is not in the database and has been added.

                    Thought: Now that the event is in the database, I will find the causal variables associated with the event and interact with the {retrieval_agent_name} agent to assess if each variable is already in the database and if so, retrieve and complete their information and use the provided names from the database to prevent duplicates.
                    The text contains two causal variables with names [VARIABLE_NAME_1], [VARIABLE_NAME_2] and properties [VARIABLE_PROPERTIES_1] and [VARIABLE_PROPERTIES_2]. Let's verify if they are in the database.
                    Code:
                    ```py
                    retrieval_agent_answer = {retrieval_agent_name}(task="Are the following variables in the database? [VARIABLE_NAME_1]: [VARIABLE_PROPERTIES_1], [VARIABLE_NAME_2]: [VARIABLE_PROPERTIES_2]")
                    print(retrieval_agent_answer)
                    ```<end_code>
                    Observation: The variable [VARIABLE_NAME_1] is in the database with the name [NEW_VARIABLE_NAME_1] and properties [NEW_VARIABLE_PROPERTIES_1]. The variable [VARIABLE_NAME_2] is not in the database and has been added. 

                    Thought: I will now update the variable name and properties and interact with the {causal_order_agent_name} agent to build a partial order from the list of causal variables.
                    Code:
                    ```py
                    order_list = {causal_order_agent_name}(task="Build a partial order from the list of causal variables", additional_args={{'causal_variables': [{{'name': '[NEW_VARIABLE_NAME_1]', , **[VARIABLE_PROPERTIES_1], **[NEW_VARIABLE_PROPERTIES_1]'}} , {{'name': '[VARIABLE_NAME_2]', **[VARIABLE_PROPERTIES_2]'}}]}})
                    ```<end_code>
                    Observation: The partial order is [(NEW_VARIABLE_NAME_1, VARIABLE_NAME_2)].

                    Thought: I will now interact with the {causal_discovery_agent_name} agent to build the full causal graph.
                    Code:
                    ```py
                    causal_graph = {causal_discovery_agent_name}(task="Build the full causal graph associated with the given variables and the partial order NEW_VARIABLE_NAME_1 < VARIABLE_NAME_2.", additional_args={{'causal_variables': [{{'name': '[NEW_VARIABLE_NAME_1]', **[NEW_VARIABLE_PROPERTIES_1]}}, {{'name': '[VARIABLE_NAME_2]', **[VARIABLE_PROPERTIES_2]}}]}})
                    final_answer(causal_graph)
                    ```<end_code>
                    """

    def __init__(self):
        self.retrieval_agent_factory = RetrievalAgentFactory()
        self.causal_order_agent_factory = CausalOrderAgentFactory()
        self.causal_discovery_agent_factory = CausalDiscoveryAgentFactory()


    def createAgent(self, base_model: Model) -> CustomSystemPromptCodeAgent:
        retrieval_agent = self.retrieval_agent_factory.createAgent(base_model)
        causal_order_agent = self.causal_order_agent_factory.createAgent(base_model)
        causal_discovery_agent = self.causal_discovery_agent_factory.createAgent(base_model)

        system_prompt = CausalDiscoveryManagerAgentFactory.SYSTEM_PROMPT.format(
            event=EVENT,
            observed_variable=OBSERVED_VARIABLE,
            variable=VARIABLE,
            causalrelationship=CAUSAL_RELATIONSHIP,
            retrieval_agent_name=retrieval_agent.name,
            causal_order_agent_name=causal_order_agent.name,
            causal_discovery_agent_name=causal_discovery_agent.name
        )

        return CustomSystemPromptCodeAgent(
                    tools=[], 
                    model=base_model, 
                    # additional_authorized_imports=["networkx"],
                    name="causal_extraction_manager_agent", 
                    description=system_prompt,
                    custom_system_prompt=system_prompt,
                    managed_agents=[retrieval_agent, causal_order_agent, causal_discovery_agent]
        )