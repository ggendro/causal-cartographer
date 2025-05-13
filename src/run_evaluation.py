import os
import argparse
import networkx as nx
import pandas as pd
import datetime
import tqdm
from typing import Generator, Tuple, Optional, List, Dict
import random
import json

from smolagents import LiteLLMModel, TransformersModel

from causal_world_modelling_agent.agents.causal_inference.causal_inference_agent import CausalInferenceAgentFactory
from causal_world_modelling_agent.world_model.world_manager import BaseWorldManager, Query, MIXING_FUNCTIONS


BALANCING_FACTORS = {
    "observation": 0.01,
    "counterfactual_match": 0.455,
    "counterfactual_mix": 0.95
}


def load_world_graph(graph_path: str) -> BaseWorldManager:
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file {graph_path} does not exist.")
    
    world_manager = BaseWorldManager(initial_graph=nx.read_gml(graph_path, destringizer=lambda x: x if x != "None" else None))
    return world_manager


def build_dataset(world_manager: BaseWorldManager, max_interventions: int, mixing_functions: str | List[str], allow_balancing: bool) -> Generator[Tuple[str, Query], None, None]:
    if isinstance(mixing_functions, str):
        if mixing_functions == "all":
            mixing_functions = MIXING_FUNCTIONS
        elif mixing_functions == "none":
            mixing_functions = []
        else:
            mixing_functions = [mixing_functions]
    
    for node in world_manager.get_complete_graph().nodes:
        for query in world_manager.generate_observations(node):
            if (not allow_balancing) or (random.random() > BALANCING_FACTORS["observation"]):
                yield "observation", query.get_dict()

        for i in range(1, max_interventions+1):
            for query in world_manager.generate_counterfactuals_match(node, num_interventions=i):
                if (not allow_balancing) or (random.random() > BALANCING_FACTORS["counterfactual_match"]):
                    yield "counterfactual_match", query.get_dict()

            if mixing_functions:
                for mixing_function in mixing_functions:
                    for query in world_manager.generate_counterfactuals_mix(node, num_interventions=i, mixing_function=mixing_function):
                        if (not allow_balancing) or (random.random() > BALANCING_FACTORS["counterfactual_mix"]**(len(mixing_functions))):
                            yield f"counterfactual_mix_{mixing_function}", query.get_dict()


def load_dataset(dataset_path: str) -> List[Dict[str, str]]:
    dataset = pd.read_csv(dataset_path)
    types = list(dataset["type"])
    dataset = dataset.drop(columns=["type"])
    dataset["causal_graph"] = dataset["causal_graph"].apply(lambda x: nx.read_gml(os.path.join(dataset_path.replace(".csv", "_graphs"), x)))
    dataset["observations"] = dataset["observations"].apply(lambda x: json.loads(x))
    dataset.loc[dataset['is_counterfactual'] == False, 'interventions'] = None
    dataset.loc[dataset['is_counterfactual'] == True, 'interventions'] = dataset.loc[dataset['is_counterfactual'] == True, 'interventions'].apply( lambda x: json.loads(x))
    dataset = dataset.to_dict(orient="records")
    dataset = [(types[i], dataset[i]) for i in range(len(dataset))]
    return dataset


def evaluate_query(query: Dict[str, str], inference_agent: CausalInferenceAgentFactory, prompt_complement: str) -> Dict[str, str]:
    query = query.copy() # Avoid modifying the original query
    query_copy = query.copy()
    query.pop("world_ids")
    query.pop("is_pseudo_gt")
    ground_truth = query.pop("ground_truth")

    causal_effect, causal_graph = inference_agent.run(prompt_complement, additional_args=query)

    is_correct = causal_effect == ground_truth
    return {
        **query_copy,
        "causal_effect": causal_effect,
        "computation_graph": json.dumps(list(causal_graph.nodes(data=True))),
        "predicted_value": causal_graph.nodes[query["target_variable"]]["current_value"],
        "correct": is_correct
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover causal structure from data.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--graph_path", type=str, help="Path to the graph file.")
    source_group.add_argument("--dataset_path", type=str, help="Path to the dataset file.")
    parser.add_argument("--model_base", type=str, default="o3-mini-2025-01-31", help="Base model to use.")
    parser.add_argument("--api_key", type=str, help="API key for the model.")
    parser.add_argument("--model_type", type=str, choices=["lite", "transformers"], default="lite", help="Type of model to use.")
    parser.add_argument("--prompt_complement", type=str, default="Compute the causal effect from the variables as required.", help="Prompt complement to help the model.") 
    parser.add_argument("--save_path", type=str, default="../data/inference", help="Path to save the inference results.")
    parser.add_argument("--dataset_save_path", type=str, default=None, help="Path to save the dataset.")
    parser.add_argument("--max_interventions", type=int, default=3, help="Maximum number of interventions to consider.")
    parser.add_argument("--mixing_functions", type=str, default="all", help="Mixing functions to use for counterfactuals.")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume the graph from a GML file.")
    parser.add_argument('--max_queries', type=int, default=None, help='Maximum number of queries to execute.')
    parser.add_argument('--allow_balancing', action='store_true', help='Allow balancing the dataset by randomly selecting queries during generation (works only with graph_path).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--build_dataset_only', action='store_true', help='Only build the dataset without running the evaluation. dataset_save_path must be specified.')
    return parser.parse_args()


def main(model_base: str, 
         api_key: str,
         model_type: str,
         prompt_complement: str,
         save_path: str, 
         dataset_save_path: str, 
         max_interventions: int, 
         mixing_functions: str, 
         graph_path: Optional[str], 
         dataset_path: Optional[str], 
         resume: Optional[str], 
         max_queries: Optional[int],
         allow_balancing: bool,
         seed: int,
         build_dataset_only: bool
         ) -> None:
    
    random.seed(seed)

    if graph_path is not None:
        world_manager = load_world_graph(graph_path)

        if ',' in mixing_functions:
            mixing_functions = mixing_functions.split(',')
        dataset = build_dataset(world_manager, max_interventions=max_interventions, mixing_functions=mixing_functions, allow_balancing=allow_balancing)
        
    elif dataset_path is not None:
        dataset = load_dataset(dataset_path)

    else:
        raise ValueError("Either graph_path or dataset_path must be provided.")

    # Load the model
    if model_type == "lite":
        base_model = LiteLLMModel(model_id=model_base, api_key=api_key)
    elif model_type == "transformers":
        import torch
        base_model = TransformersModel(model_id=model_base, device_map="cuda:0" if torch.cuda.is_available() else "cpu", max_new_tokens=4000)
    else:
        raise ValueError("Invalid model type. Choose either 'lite' or 'transformers'.")
    
    # Create the inference agent
    inference_agent = CausalInferenceAgentFactory().createAgent(base_model)

    # Evaluate on observations and counterfactuals
    if resume:
        results = pd.read_csv(resume)
        skip_rows = len(results)
        print(f"Resuming from {resume}. {skip_rows} rows already processed.")
    else:
        results = pd.DataFrame(columns=list(Query.__annotations__.keys()) + ["type", "causal_effect", "predicted_value", "computation_graph", "correct", "error"])
        skip_rows = 0

    if dataset_save_path:
        os.makedirs(dataset_save_path, exist_ok=True)
        os.makedirs(os.path.join(dataset_save_path, "graphs_temp"), exist_ok=False)
        dataset_save = pd.DataFrame(columns=list(Query.__annotations__.keys()) + ["type"])

    skips = 0
    try:
        for i, (query_type, query) in enumerate(tqdm.tqdm(dataset)):
            if skips < skip_rows:
                skips += 1
                continue

            if max_queries and i >= (max_queries + skip_rows):
                print(f"Processed {max_queries} documents. Stopping.")
                break

            if dataset_save_path: # save the dataset if specified (must be done in the for loop as the dataset is a generator)
                instance_save = {
                    **query,
                    "type": query_type
                }
                nx.write_gml(instance_save["causal_graph"], os.path.join(dataset_save_path, "graphs_temp", f"graph_{i}.gml"), stringizer=str)
                instance_save["causal_graph"] = f"graph_{i}.gml"
                dataset_save.loc[len(dataset_save)] = instance_save

            if build_dataset_only:
                continue

            try:
                query_res = evaluate_query(query, inference_agent, prompt_complement)
                results.loc[len(results)] = {
                    **query_res,
                    "type": query_type,
                    "error": None
                }

            except Exception as e:
                print(f"Error processing query: {query}. Error: {e}")
                results.loc[len(results)] = {
                    **query,
                    "causal_effect": None,
                    "computation_graph": None,
                    "predicted_value": None,
                    "correct": False,
                    "type": query_type,
                    "error": str(e)
                }

    except KeyboardInterrupt:
        print("Evaluation interrupted by user.")
        
    # Save results to CSV
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not build_dataset_only:
        os.makedirs(save_path, exist_ok=True)
        results.to_csv(os.path.join(save_path, f"evaluation_results_{current_time}.csv"), index=False)

    if dataset_save_path:
        dataset_save["observations"] = dataset_save["observations"].apply(lambda x: json.dumps(x)) # Convert to JSON string
        dataset_save["interventions"] = dataset_save["interventions"].apply(lambda x: json.dumps(x))
        os.makedirs(dataset_save_path, exist_ok=True)
        dataset_save.to_csv(os.path.join(dataset_save_path, f"dataset_{current_time}.csv"), index=False)
        os.rename(os.path.join(dataset_save_path, "graphs_temp"), os.path.join(dataset_save_path, f"dataset_{current_time}_graphs"))



if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
