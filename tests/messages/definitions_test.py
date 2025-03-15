
import pytest

from causal_world_modelling_agent.syntax.definitions import EventDefinition, VariableDefinition, CausalRelationshipDefinition


class TestEventDefinition:

    def test_from_dict(self):
        event_dict = {
            "name": "event_name",
            "description": "event_description",
            "time": "event_time",
            "location": "event_location",
            "variables": "event_variables"
        }
        event = EventDefinition.from_dict(event_dict)
        assert event.name == "event_name"
        assert event.description == "event_description"
        assert event.time == "event_time"
        assert event.location == "event_location"
        assert event.variables == "event_variables"

    def test_from_dict_with_missing_keys(self):
        event_dict = {
            "name": "event_name",
            "description": "event_description",
            "time": "event_time",
            "location": "event_location",
        }
        pytest.raises(TypeError, EventDefinition.from_dict, event_dict)

    def test_from_dict_with_extra_keys(self):
        event_dict = {
            "name": "event_name",
            "description": "event_description",
            "time": "event_time",
            "location": "event_location",
            "variables": "event_variables",
            "extra_key": "extra_value"
        }
        pytest.raises(TypeError, EventDefinition.from_dict, event_dict)

    @pytest.fixture
    def event(self):
        return EventDefinition(
            name="event_name",
            description="event_description",
            time="event_time",
            location="event_location",
            variables="event_variables"
        )
    
    def test_to_dict(self, event):
        event_dict = event.to_dict()
        assert event_dict["name"] == "event_name"
        assert event_dict["description"] == "event_description"
        assert event_dict["time"] == "event_time"
        assert event_dict["location"] == "event_location"
        assert event_dict["variables"] == "event_variables"



class TestCausalReationshipDefinition:
    
    def test_from_dict(self):
        causal_relationship_dict = {
            "cause": "cause_name",
            "effect": "effect_name",
            "description": "causal_relationship_description",
            "contextual_information": "contextual_information",
            "type": "causal_relationship_type",
            "strength": "causal_relationship_strength",
            "confidence": "causal_relationship_confidence",
            "function": "causal_relationship_function"
        }
        causal_relationship = CausalRelationshipDefinition.from_dict(causal_relationship_dict)
        assert causal_relationship.cause == "cause_name"
        assert causal_relationship.effect == "effect_name"
        assert causal_relationship.description == "causal_relationship_description"
        assert causal_relationship.contextual_information == "contextual_information"
        assert causal_relationship.type == "causal_relationship_type"
        assert causal_relationship.strength == "causal_relationship_strength"
        assert causal_relationship.confidence == "causal_relationship_confidence"
        assert causal_relationship.function == "causal_relationship_function"

    def test_from_dict_with_missing_keys(self):
        causal_relationship_dict = {
            "description": "causal_relationship_description",
            "contextual_information": "contextual_information",
            "type": "causal_relationship_type",
            "strength": "causal_relationship_strength",
            "confidence": "causal_relationship_confidence",
            "function": "causal_relationship_function"
        }
        pytest.raises(TypeError, CausalRelationshipDefinition.from_dict, causal_relationship_dict)

    def test_from_dict_with_missing_function(self):
        causal_relationship_dict = {
            "cause": "cause_name",
            "effect": "effect_name",
            "description": "causal_relationship_description",
            "contextual_information": "contextual_information",
            "type": "causal_relationship_type",
            "strength": "causal_relationship_strength",
            "confidence": "causal_relationship_confidence",
        }
        causal_relationship = CausalRelationshipDefinition.from_dict(causal_relationship_dict)
        assert causal_relationship.function == None
        