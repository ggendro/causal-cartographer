


EVENT = """
        {
            "name": <string>, # The name of the event
            "description": <string>, # The description of the event
            "time": <string>, # The time period of the event
            "location": <string>, # The location of the event
            "variables": <list>, # The variable names that are involved in the event
        }
        """

VARIABLE = """
        {
            "name": <string>, # The name of the variable
            "description": <string>, # The description of the variable
            "type": <string>, # The type of the variable (boolean, integer, float, string, etc.)
            "values": <list>, # The set or range of possible values for the variable ([1, 2, 3], 'range(0,10)', ['low', 'medium', 'high'], 'True/False', 'natural numbers', etc.)
        }
        """

OBSERVED_VARIABLE = """
        {
            "name": <string>, # The name of the variable
            "description": <string>, # The description of the variable
            "type": <string>, # The type of the variable (boolean, integer, float, string, etc.)
            "values": <list>, # The set or range of possible values for the variable
            "current_value": <string>, # The observed current value of the variable
            "contextual_information": <string>, # The contextual information associated with the current value of the variable
        }
        """

CAUSAL_RELATIONSHIP = """
        {
            "cause": <string>, # The name of the cause variable
            "effect": <string>, # The name of the effect variable
            "description": <string>, # The description of the causal relationship between the variables
            "contextual_information": <string>, # The contextual information associated with the causal relationship for the specific observed values of the variables
            "type": <string>, # The type of the causal relationship (direct, indirect, etc.)
            "strength": <string>, # The strength of the causal relationship
            "confidence": <string>, # The confidence level in the existence of the causal relationship
            "function": <Optional[Callable]>, # The function that describes the causal relationship, if available.
        }
        """