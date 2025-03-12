
from typing import List, Dict, Optional

from smolagents import ChatMessage, Model, Tool



class MockModel(Model):
    """
    Mock model for testing purposes. Returns the source code provided in the constructor.
    """
    

    def __init__(self, model_id: str, source_code: str):
        self.model_id = model_id
        self.source_code = source_code

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        return ChatMessage(
            role="assistant",
            content=self.source_code
        )
    


class DummyMockModel(MockModel):
    """
    Mock model for testing purposes. Returns a dummy source code.
    """
    SOURCE_CODE = """
```
final_answer("Dummy answer!")
```<end_code>
"""

    def __init__(self):
        super().__init__(model_id="dummy_mock_model", source_code=self.SOURCE_CODE)


class UpdateGraphMockModel(MockModel):
    """
    Mock model for causal discovery testing. Updates the graph given the input message.
    Assumes that the graph is accessible as a parameter `G`.
    """

    SOURCE_CODE = """
```
G.add_node("node1")
G.add_node("node2")
G.add_edge("node1", "node2")
final_answer(G)
```<end_code>
"""

    def __init__(self):
        super().__init__(model_id="update_graph_mock_model", source_code=self.SOURCE_CODE)


class CountInferenceMockModel(MockModel):
    """
    Mock model for causal inference testing. Updates the value of a child variable given its parent variables.
    Assumes that the parent variables are accessible as a parameter `parent_variables` and the child variable is accessible as a parameter `target_variable`.
    """

    SOURCE_CODE = """
```
count = 0
for parent in {parent_variables}:
    count += parent["causal_effect"] * ({reverse})
target_variable["causal_effect"] = count + 1
final_answer(target_variable)
```<end_code>
"""
    
    def __init__(self):
        super().__init__(model_id="count_inference_mock_model", source_code=self.SOURCE_CODE)

    def __call__(
        self,
        messages: List[Dict[str, str]],
        stop_sequences: Optional[List[str]] = None,
        grammar: Optional[str] = None,
        tools_to_call_from: Optional[List[Tool]] = None,
        **kwargs,
    ) -> ChatMessage:
        if "parent_variables" in messages[-1]['content'][0]['text']:
            self.source_code = self.SOURCE_CODE.format(parent_variables="parent_variables", reverse=1)
        else:
            self.source_code = self.SOURCE_CODE.format(parent_variables="children_variables", reverse=-1)
        return super().__call__(messages, stop_sequences, grammar, tools_to_call_from, **kwargs)