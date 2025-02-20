from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from smolagents import tool

from ..core.definitions import Message


class Database(ABC):

    @abstractmethod
    def findEvent(self, event_name: str) -> Optional[Message]:
        """
        Find an event in the database.
        
        Args:
            event_name: The name of the event to find.
        
        Returns:
            The content of the event if found, None otherwise.
        """
        pass

    @abstractmethod
    def listEvents(self) -> List[str]:
        """
        List all event names in the database.
        
        Returns:
            A list of all event names in the database.
        """
        pass

    @abstractmethod
    def addEvent(self, event_name: str, event: Message) -> bool:
        """
        Add an event to the database.
        
        Args:
            event_name: The name of the event to add.
            event: The content of the event to add.
        
        Returns:
            True if the event was added, False otherwise.
        """
        pass

    @abstractmethod
    def removeEvent(self, event_name: str) -> bool:
        """
        Remove an event from the database.
        
        Args:
            event_name: The name of the event to remove.
        
        Returns:
            True if the event was removed, False otherwise.
        """
        pass

    @abstractmethod
    def editEvent(self, event_name: str, event: Message) -> bool:
        """
        Edit an event in the database.
        
        Args:
            event_name: The name of the event to edit.
            event: The content of the event to edit.
        
        Returns:
            True if the event was edited, False otherwise.
        """
        pass

    @abstractmethod
    def findCausalVariable(self, variable_name: str) -> Optional[Message]:
        """
        Find a causal variable in the database.
        
        Args:
            variable_name: The name of the causal variable to find.
        
        Returns:
            The content of the causal variable if found, None otherwise.
        """
        pass

    @abstractmethod
    def listCausalVariables(self) -> List[str]:
        """
        List all causal variable names in the database.
        
        Returns:
            A list of all causal variable names in the database.
        """
        pass

    @abstractmethod
    def addCausalVariable(self, variable_name: str, variable: Message) -> bool:
        """
        Add a causal variable to the database.
        
        Args:
            variable_name: The name of the causal variable to add.
            variable: The content of the causal variable to add.
        
        Returns:
            True if the causal variable was added, False otherwise.
        """
        pass

    @abstractmethod
    def removeCausalVariable(self, variable_name: str) -> bool:
        """
        Remove a causal variable from the database.
        
        Args:
            variable_name: The name of the causal variable to remove.
        
        Returns:
            True if the causal variable was removed, False otherwise.
        """
        pass

    @abstractmethod
    def editCausalVariable(self, variable_name: str, variable: Message) -> bool:
        """
        Edit a causal variable in the database.
        
        Args:
            variable_name: The name of the causal variable to edit.
            variable: The content of the causal variable to edit.
        
        Returns:
            True if the causal variable was edited, False otherwise.
        """
        pass

    @abstractmethod
    def findCausalRelationship(self, source_variable: str, target_variable: str) -> Optional[Message]:
        """
        Find a causal relationship in the database.
        
        Args:
            source_variable: The name of the source variable of the relationship to find.
            target_variable: The name of the target variable of the relationship to find.
        
        Returns:
            The content of the causal relationship if found, None otherwise.
        """
        pass

    @abstractmethod
    def listCausalRelationships(self) -> List[Tuple[str, str]]:
        """
        List all causal relationships in the database.

        Returns:
            A list of tuples containing the source and target variable names of the causal relationships in the database.
        """
        pass

    @abstractmethod
    def addCausalRelationship(self, source_variable: str, target_variable: str, relationship: Message) -> bool:
        """
        Add a causal relationship to the database.
        
        Args:
            source_variable: The name of the source variable of the relationship to add.
            target_variable: The name of the target variable of the relationship to add.
            relationship: The content of the relationship to add.
        
        Returns:
            True if the causal relationship was added, False otherwise.
        """
        pass

    @abstractmethod
    def removeCausalRelationship(self, source_variable: str, target_variable: str) -> bool:
        """
        Remove a causal relationship from the database.
        
        Args:
            source_variable: The name of the source variable of the relationship to remove.
            target_variable: The name of the target variable of the relationship to remove.
        
        Returns:
            True if the causal relationship was removed, False otherwise.
        """
        pass

    @abstractmethod
    def editCausalRelationship(self, source_variable: str, target_variable: str, relationship: Message) -> bool:
        """
        Edit a causal relationship in the database.
        
        Args:
            source_variable: The name of the source variable of the relationship to edit.
            target_variable: The name of the target variable of the relationship to edit.
            relationship: The content of the relationship to edit.
        
        Returns:
            True if the causal relationship was edited, False otherwise.
        """
        pass


class SimpleJsonDatabase:
    
    def __init__(self):
        self.events = {}
        self.variables = {}
        self.relationships = {}

    def findEvent(self, event_name: str) -> Optional[Message]:
        return self.events.get(event_name, None)
    
    def listEvents(self) -> List[str]:
        return list(self.events.keys())
    
    def addEvent(self, event_name: str, event: Message) -> bool:
        if event_name in self.events:
            return False
        self.events[event_name] = event
        return True
    
    def removeEvent(self, event_name: str) -> bool:
        if event_name not in self.events:
            return False
        del self.events[event_name]
        return True
    
    def editEvent(self, event_name: str, event: Message) -> bool:
        if event_name not in self.events:
            return False
        self.events[event_name] = event
        return True
    
    def findCausalVariable(self, variable_name: str) -> Optional[Message]:
        return self.variables.get(variable_name, None)
    
    def listCausalVariables(self) -> List[str]:
        return list(self.variables.keys())
    
    def addCausalVariable(self, variable_name: str, variable: Message) -> bool:
        if variable_name in self.variables:
            return False
        self.variables[variable_name] = variable
        return True
    
    def removeCausalVariable(self, variable_name: str) -> bool:
        if variable_name not in self.variables:
            return False
        del self.variables[variable_name]
        return True
    
    def editCausalVariable(self, variable_name: str, variable: Message) -> bool:
        if variable_name not in self.variables:
            return False
        self.variables[variable_name] = variable
        return True
    
    def findCausalRelationship(self, source_variable: str, target_variable: str) -> Optional[Message]:
        return self.relationships.get((source_variable, target_variable), None)
    
    def listCausalRelationships(self) -> List[Tuple[str, str]]:
        return list(self.relationships.keys())
    
    def addCausalRelationship(self, source_variable: str, target_variable: str, relationship: Message) -> bool:
        if (source_variable, target_variable) in self.relationships:
            return False
        self.relationships[(source_variable, target_variable)] = relationship
        return True
    
    def removeCausalRelationship(self, source_variable: str, target_variable: str) -> bool:
        if (source_variable, target_variable) not in self.relationships:
            return False
        del self.relationships[(source_variable, target_variable)]
        return True
    
    def editCausalRelationship(self, source_variable: str, target_variable: str, relationship: Message) -> bool:
        if (source_variable, target_variable) not in self.relationships:
            return False
        self.relationships[(source_variable, target_variable)] = relationship
        return True


@tool
def findEvent(database: Database, event_name: str) -> Optional[Message]:
    """
    Find an event in the database.

    Args:
        database: The database object.
        event_name: The name of the event to find.
    
    Returns:
        The content of the event if found, None otherwise.
    """
    return database.findEvent(event_name)

@tool
def listEvents(database: Database) -> List[str]:
    """
    List all event names in the database.
    
    Args:
        database: The database object.
    
    Returns:
        A list of all event names in the database.
    """
    return database.listEvents()

@tool
def addEvent(database: Database, event_name: str, event: Message) -> bool:
    """
    Add an event to the database.
    
    Args:
        database: The database object.
        event_name: The name of the event to add.
        event: The content of the event to add.
    
    Returns:
        True if the event was added, False otherwise.
    """
    return database.addEvent(event_name, event)

@tool
def removeEvent(database: Database, event_name: str) -> bool:
    """
    Remove an event from the database.
    
    Args:
        database: The database object.
        event_name: The name of the event to remove.
    
    Returns:
        True if the event was removed, False otherwise.
    """
    return database.removeEvent(event_name)

@tool
def editEvent(database: Database, event_name: str, event: Message) -> bool:
    """
    Edit an event in the database.
    
    Args:
        database: The database object.
        event_name: The name of the event to edit.
        event: The content of the event to edit.
    
    Returns:
        True if the event was edited, False otherwise.
    """
    return database.editEvent(event_name, event)

@tool
def findCausalVariable(database: Database, variable_name: str) -> Optional[Message]:
    """
    Find a causal variable in the database.
    
    Args:
        database: The database object.
        variable_name: The name of the causal variable to find.
    
    Returns:
        The content of the causal variable if found, None otherwise.
    """
    return database.findCausalVariable(variable_name)

@tool
def listCausalVariables(database: Database) -> List[str]:
    """
    List all causal variable names in the database.
    
    Args:
        database: The database object.
    
    Returns:
        A list of all causal variable names in the database.
    """
    return database.listCausalVariables()

@tool
def addCausalVariable(database: Database, variable_name: str, variable: Message) -> bool:
    """
    Add a causal variable to the database.
    
    Args:
        database: The database object.
        variable_name: The name of the causal variable to add.
        variable: The content of the causal variable to add.
    
    Returns:
        True if the causal variable was added, False otherwise.
    """
    return database.addCausalVariable(variable_name, variable)

@tool
def removeCausalVariable(database: Database, variable_name: str) -> bool:
    """
    Remove a causal variable from the database.
    
    Args:
        database: The database object.
        variable_name: The name of the causal variable to remove.
    
    Returns:
        True if the causal variable was removed, False otherwise.
    """
    return database.removeCausalVariable(variable_name)

@tool
def editCausalVariable(database: Database, variable_name: str, variable: Message) -> bool:
    """
    Edit a causal variable in the database.
    
    Args:
        database: The database object.
        variable_name: The name of the causal variable to edit.
        variable: The content of the causal variable to edit.
    
    Returns:
        True if the causal variable was edited, False otherwise.
    """
    return database.editCausalVariable(variable_name, variable)

@tool
def findCausalRelationship(database: Database, source_variable: str, target_variable: str) -> Optional[Message]:
    """
    Find a causal relationship in the database.
    
    Args:
        database: The database object.
        source_variable: The name of the source variable of the relationship to find.
        target_variable: The name of the target variable of the relationship to find.
    
    Returns:
        The content of the causal relationship if found, None otherwise.
    """
    return database.findCausalRelationship(source_variable, target_variable)

@tool
def listCausalRelationships(database: Database) -> List[Tuple[str, str]]:
    """
    List all causal relationships in the database.
    
    Args:
        database: The database object.
    
    Returns:
        A list of tuples containing the source and target variable names of the causal relationships in the database.
    """
    return database.listCausalRelationships()

@tool
def addCausalRelationship(database: Database, source_variable: str, target_variable: str, relationship: Message) -> bool:
    """
    Add a causal relationship to the database.
    
    Args:
        database: The database object.
        source_variable: The name of the source variable of the relationship to add.
        target_variable: The name of the target variable of the relationship to add.
        relationship: The content of the relationship to add.
    
    Returns:
        True if the causal relationship was added, False otherwise.
    """
    return database.addCausalRelationship(source_variable, target_variable, relationship)

@tool
def removeCausalRelationship(database: Database, source_variable: str, target_variable: str) -> bool:
    """
    Remove a causal relationship from the database.
    
    Args:
        database: The database object.
        source_variable: The name of the source variable of the relationship to remove.
        target_variable: The name of the target variable of the relationship to remove.
    
    Returns:
        True if the causal relationship was removed, False otherwise.
    """
    return database.removeCausalRelationship(source_variable, target_variable)

@tool
def editCausalRelationship(database: Database, source_variable: str, target_variable: str, relationship: Message) -> bool:
    """
    Edit a causal relationship in the database.
    
    Args:
        database: The database object.
        source_variable: The name of the source variable of the relationship to edit.
        target_variable: The name of the target variable of the relationship to edit.
        relationship: The content of the relationship to edit.
    
    Returns:
        True if the causal relationship was edited, False otherwise.
    """
    return database.editCausalRelationship(source_variable, target_variable, relationship)
