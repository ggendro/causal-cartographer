
from typing import Optional, Dict

from smolagents import Model

from ...core.agent import CustomSystemPromptCodeAgent
from ..factory import AgentFactory
from ...tools.database import (
    SimpleJsonDatabase, 
    findEvent,
    listEvents,
    addEvent,
    removeEvent,
    editEvent,
    findCausalVariable,
    listCausalVariables,
    addCausalVariable,
    removeCausalVariable,
    editCausalVariable,
    findCausalRelationship,
    listCausalRelationships,
    addCausalRelationship,
    removeCausalRelationship,
    editCausalRelationship
)
from ...tools.wikifier import findCorrespondingWikiDataConcept
from ...syntax.messages import EVENT, OBSERVED_VARIABLE, CAUSAL_RELATIONSHIP




class DatabaseAgent(CustomSystemPromptCodeAgent):

    def __init__(self, *args, database: Optional[SimpleJsonDatabase] = None, **kwargs):
        super().__init__(*args, **kwargs)
        
        if not database:
            raise ValueError("Database must be provided")
        self.database = database

    def run(self, *args, additional_args: Optional[Dict] = None, **kwargs):
        if additional_args:
            additional_args["database"] = self.database
        else:
            additional_args = {"database": self.database}

        super().run(*args, additional_args=additional_args, **kwargs)



class RetrievalAgentFactory(AgentFactory):

    AGENT_NAME = "retrieve_and_store_agent"

    DESCRIPTION = """Agent that communicates with a database to store and retrieve events, variables and relationships. 
                    Given a event, the agent can retrieve additional variables and relationships linked to the event. The agent can also ensure that the database does not contain duplicates variables by finding, given a description, if a similar varaible already exists in the database.  
                    Example of agent call to check if an event is in the database:
                    ```
                    agent_answer = {agent_name}(task="Is the following event in the database? {{...}}")
                    print(agent_answer) # 'Yes it is. here are additional variables associated with this event: [{{...}}]' or 'No it was not. I added it to the database.' 
                    ```
                    Example of agent call to check if a variable is in the database:
                    ```
                    agent_answer = {agent_name}(task="Is the following variable in the database? {{...}}")
                    print(agent_answer) # 'Yes it is. Here is the already stored variable: {{...}} to use.' or 'No it was not. I added it to the database.'
                    ```
                    Example of agent call to add a relationship to the database:
                    ```
                    agent_answer = {agent_name}(task="Add the following relationship to the database: {{...}}")
                    print(agent_answer) # 'I added the relationship to the database.' or 'The relationship was already in the database.'
                    """

    SYSTEM_PROMPT = f"""You are an agent that communicates with a database to store and retrieve events, variables and relationships.
                    The database is provided as an argument with name 'database'.
                    The agent must add to the database any new event, variable or relationship that it is being sent.
                    The agent must not directly add variables to the database, instead, it must find the closest wikidata concept to the variable and add it to the database.
                    When asked, it should always return the wikidata concept of a variable instead of the variable itself.
                    Events have the following structure:
                    {EVENT}
                    Variables have the following structure:
                    {OBSERVED_VARIABLE}
                    Causal Relationships have the following structure:
                    {CAUSAL_RELATIONSHIP}
                    """

    def createAgent(self, base_model: Model) -> DatabaseAgent:
        return DatabaseAgent(
                    tools=[
                        findEvent,
                        listEvents,
                        addEvent,
                        removeEvent,
                        editEvent,
                        findCausalVariable,
                        listCausalVariables,
                        addCausalVariable,
                        removeCausalVariable,
                        editCausalVariable,
                        findCausalRelationship,
                        listCausalRelationships,
                        addCausalRelationship,
                        removeCausalRelationship,
                        editCausalRelationship,
                        findCorrespondingWikiDataConcept
                    ],
                    model=base_model,
                    name=RetrievalAgentFactory.AGENT_NAME,
                    description=RetrievalAgentFactory.DESCRIPTION.format(agent_name=RetrievalAgentFactory.AGENT_NAME),
                    custom_system_prompt=RetrievalAgentFactory.SYSTEM_PROMPT,
                    database=SimpleJsonDatabase()
        )