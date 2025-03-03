# A Causal World Modelling Agent


<img src="assets/dialog-example.png" alt="Example of Agent Dialog" width="1200"/>


## Introduction

Repository for the Causal World Modelling Agent project. This framework is an LLM-based approach for the extraction of causal graphs from natural language text and composition of causal world models. The approach builds a causal model and verifies its consistency across counterfactual scenarios built from multiples sources.


## Installation


```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

On Windows, use `env\Scripts\activate` instead of `source env/bin/activate`.

A subscription to an LLM API provider may be required.


## Usage

TODO

For now, you can run the `smolagents_notebook.ipynb` Jupyter notebook available at [src/smolagents_notebook.ipynb](src/smolagents_notebook.ipynb).

<!-- ## Messages


### Standard Messages

Agents communicate via messages. A message is a dictionary with the following keys:
```python
{
    "role": "role_name", # The role of the agent that sends the message. Can be 'user' or 'assistant'.
    "content": "content",
}
```
This structure follows the messages syntax from the [OpenAI API](https://platform.openai.com/docs/guides/text-generation).


### Tool Calls

A tool call is a message that is sent to a tool. The message is a dictionary with the following keys:
```python
{
    "type": "function",
    "function": {
        "name": "tool_name",
        "description": "A description of the tool."
        "parameters": {
            "type": "object",
            "properties": {
                "parameter_name_1": {
                    "type": "type",
                    "description": "A description of the parameter."
                }
                "parameter_name_N": {
                    "type": "type",
                    "description": "A description of the parameter."
                }
            }
        }
    }
}
```
This structure follows the tools syntax from the [OpenAI API](https://platform.openai.com/docs/assistants/tools/function-calling). -->


## List of Agents

| Agent Name | Description | Standalone | Component | Dependencies | Implementation |
|------------|-------------|------------|-----------|--------------|----------------|
| `end_to_end_causal_extraction_agent` | Agent that builds a simple networkx causal graph from a text snippet using pre-trained commonsense knowledge. | :white_check_mark: | :white_check_mark: | `networkx` | :white_check_mark: |
| `causal_inference_agent` | Agent that performs counterfactual causal inference on a networkx causal graph using an LLM as an inference engine. | :white_check_mark: | :white_check_mark: | | :white_check_mark: |
| `causal_inference_evaluation_agent` | Agent that builds counterfactual queries and evaluate a networkx causal graph using multiple evaluation methods. | :white_check_mark: |  | `causal_inference_agent` | TODO |
| `causal_extraction_manager_agent` | Agent that builds a networkx causal graph from a text snippet and simulataneously builds a json database. Works by making calls to other agents. | :white_check_mark: | | `retrieval_agent`, `causal_order_agent`, `causal_discovery_agent` | In progress (deprecated) |
| `retrieval_agent` | Agent that retrieves variables and relationships from a json database. |  | :white_check_mark: | | In progress (deprecated) |
| `causal_order_agent` | Agent that orders the variables in a causal graph. |  | :white_check_mark: | `is_a_valid_partial_order` | In progress (deprecated) |
| `causal_discovery_agent` | Agent that discovers the relationships between variables in a causal graph. |  | :white_check_mark: | `findEvent`, `listEvents`, `addEvent`, `removeEvent`, `editEvent`, `findCausalVariable`, `listCausalVariables`, `addCausalVariable`, `removeCausalVariable`, `editCausalVariable`, `findCausalRelationship`, `listCausalRelationships`, `addCausalRelationship`, `removeCausalRelationship`, `editCausalRelationship`, `findCorrespondingWikiDataConcept` | In progress (deprecated) |
| `self_iterative_agent` | Agent that recursively builds a networkx causal graph from a text snippet or a topic. Stops on after a predefined number of steps. Self-discovery of causal graphs.  | :white_check_mark: | | `networkx` | TODO |
| `causal_extraction_rag_manager_agent` | Agent that builds a networkx causal graph from a text snippet using a RAG model. | :white_check_mark: | |  | TODO |


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments









