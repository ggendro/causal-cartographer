
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
        d = self.__dict__
        for key in list(d.keys()):
            if d[key] is None:
                del d[key]
        return d
    

@dataclass
class EventDefinition(MessageDefinition): # TODO: assess if needs to be removed
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
class _NodeDefinition(MessageDefinition):
    name: str
    description: str
    type: str
    values: List[str]

@dataclass
class VariableDefinition(_NodeDefinition):
    """{
        "name": <string>, # The name of the variable
        "description": <string>, # The description of the variable
        "type": <string>, # The type of the variable (boolean, integer, float, string, etc.)
        "values": <List[str]>, # The set or range of possible values for the variable ([1, 2, 3], 'range(0,10)', ['low', 'medium', 'high'], 'True/False', 'natural numbers', etc.)
        "supporting_text_snippets": <Optional[List[str]]>, # The supporting text snippets in which the variable is mentioned
        "current_value": <Optional[string]>, # The observed current value of the variable
        "contextual_information": <Optional[string]>, # The contextual information associated with the current value of the variable
    }"""
    supporting_text_snippets: Optional[List[str]] = None
    current_value: Optional[str] = None
    contextual_information: Optional[str] = None


@dataclass
class _CausalEffectDefinition(MessageDefinition):
    causal_effect: str

@dataclass
class InferredVariableDefinition(VariableDefinition, _CausalEffectDefinition):
    """{
        "name": <string>, # The name of the variable
        "description": <string>, # The description of the variable
        "type": <string>, # The type of the variable (boolean, integer, float, string, etc.)
        "values": <List[str]>, # The set or range of possible values for the variable ([1, 2, 3], 'range(0,10)', ['low', 'medium', 'high'], 'True/False', 'natural numbers', etc.)
        "causal_effect": <string>, # The inferred causal effect of the variable
        "supporting_text_snippets": <Optional[List[str]]>, # The supporting text snippets in which the variable is mentioned
        "current_value": <Optional[string]>, # The observed current value of the variable
        "contextual_information": <Optional[string]>, # The contextual information associated with the current value of the variable
    }"""

@dataclass
class CausalRelationshipDefinition(MessageDefinition):
    """{
        "cause": <string>, # The name of the cause variable
        "effect": <string>, # The name of the effect variable
        "description": <string>, # The description of the causal relationship between the variables
        "contextual_information": <Optional[string]>, # The contextual information associated with the causal relationship for the specific observed values of the variables
        "type": <string>, # The type of the causal relationship (direct, indirect, etc.)
        "strength": <Optional[string]>, # The strength of the causal relationship
        "confidence": <Optional[string]>, # The confidence level in the existence of the causal relationship
        "function": <Optional[Callable]>, # The function that describes the causal relationship, if available.
    }"""
    cause: str
    effect: str
    description: str
    type: str
    contextual_information: Optional[str ] = None
    strength: Optional[str] = None
    confidence: Optional[str] = None
    function: Optional[Callable] = None