
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
import inspect


Message = Dict[str, str]


@dataclass
class MessageDefinition:
    
    @classmethod
    def get_definition(cls) -> str:
        return inspect.getdoc(cls)

    @classmethod
    def from_dict(cls, data: Message) -> 'MessageDefinition':
        return cls(**data)
    
    def to_dict(self) -> Message:
        return self.__dict__
    

@dataclass
class EventDefinition(MessageDefinition):
    """{
        "name": <string>, # The name of the event
        "description": <string>, # The description of the event
        "time": <string>, # The time period of the event
        "location": <string>, # The location of the event
        "variables": <list>, # The variable names that are involved in the event
    }"""
    name: str
    description: str
    time: str
    location: str
    variables: List[str]

@dataclass
class VariableDefinition(MessageDefinition):
    """{
        "name": <string>, # The name of the variable
        "description": <string>, # The description of the variable
        "type": <string>, # The type of the variable (boolean, integer, float, string, etc.)
        "values": <list>, # The set or range of possible values for the variable ([1, 2, 3], 'range(0,10)', ['low', 'medium', 'high'], 'True/False', 'natural numbers', etc.)
        "supporting_text_snippets": <list>, # The supporting text snippets in which the variable is mentioned
    }"""
    name: str
    description: str
    type: str
    values: List[str]
    supporting_text_snippets: List[str]

@dataclass
class ObservedVariableDefinition(MessageDefinition):
    """{
        "name": <string>, # The name of the variable
        "description": <string>, # The description of the variable
        "type": <string>, # The type of the variable (boolean, integer, float, string, etc.)
        "values": <list>, # The set or range of possible values for the variable
        "supporting_text_snippets": <list>, # The supporting text snippets in which the variable is mentioned
        "current_value": <string>, # The observed current value of the variable
        "contextual_information": <string>, # The contextual information associated with the current value of the variable
    }"""
    name: str
    description: str
    type: str
    values: List[str]
    supporting_text_snippets: List[str]
    current_value: str
    contextual_information: str

@dataclass
class CausalRelationshipDefinition(MessageDefinition):
    """{
        "cause": <string>, # The name of the cause variable
        "effect": <string>, # The name of the effect variable
        "description": <string>, # The description of the causal relationship between the variables
        "contextual_information": <string>, # The contextual information associated with the causal relationship for the specific observed values of the variables
        "type": <string>, # The type of the causal relationship (direct, indirect, etc.)
        "strength": <string>, # The strength of the causal relationship
        "confidence": <string>, # The confidence level in the existence of the causal relationship
        "function": <Optional[Callable]>, # The function that describes the causal relationship, if available.
    }"""
    cause: str
    effect: str
    description: str
    contextual_information: str
    type: str
    strength: str
    confidence: str
    function: Optional[Callable] = None