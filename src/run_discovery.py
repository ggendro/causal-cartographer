
import os
import json
import argparse
from typing import Generator
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import tqdm

from smolagents import LiteLLMModel

from causal_world_modelling_agent.agents.causal_discovery.atomic_rag_agent import AtomicRAGDiscoveryAgentFactory
from causal_world_modelling_agent.world_model.world_manager import BaseWorldManager



def load_data(data_dir: str) -> Generator[str, None, None]:
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            with open(os.path.join(data_dir, filename)) as f:
                json_file = json.load(f)
                str_file = f"Title: {json_file['title']['eng']}\n - \nContent:\n{json_file['summary']}"
                yield str_file

def data_len(data_dir: str) -> int:
    count = 0
    for filename in os.listdir(data_dir):
        if filename.endswith('.json'):
            count += 1
    return count


def display_plot(causal_graph: nx.Graph, save_path: str = None) -> None:
    # Get partitions
    partitions = list(nx.algorithms.community.louvain_communities(causal_graph.to_undirected()))

    # Create a color map: one color per partition
    colors = [plt.cm.tab20(i % 20) for i in range(len(partitions))]

    # Create clustered graph layout
    center = list(nx.spring_layout(causal_graph, scale=5, seed=42).values())

    pos = {}
    for center, partition in zip(center, partitions):
        pos.update(nx.spring_layout(causal_graph.subgraph(partition), center=center))

    # Plot graph
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_edges(causal_graph, pos, alpha=0.5)

    # Draw nodes with the color of their partition
    for i, partition in enumerate(partitions):
        nx.draw_networkx_nodes(causal_graph, pos, nodelist=partition, node_color=[colors[i]], node_size=100, alpha=0.7)
        centroid = nx.center(causal_graph.subgraph(partition).to_undirected())
        if len(centroid) > 0:
            plt.text(*pos[centroid[0]], s=centroid[0], 
                    bbox=dict(facecolor='white', alpha=0.5), clip_on=True)
            
    # Plot or save graph
    if save_path:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        os.makedirs(save_path, exist_ok=True)
        nx.write_gml(causal_graph, os.path.join(save_path, f"causal_graph_{current_time}.gml"), stringizer=str)
        plt.savefig(os.path.join(save_path, f"causal_graph_{current_time}.png"), bbox_inches='tight')
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover causal structure from data.")
    parser.add_argument("--data_dir", type=str, default="../data/data", help="Directory containing JSON files to load.")
    parser.add_argument("--model_base", type=str, default="o3-mini-2025-01-31", help="Base model to use.")
    parser.add_argument("--api_key", type=str, help="API key for the LLM model.")
    parser.add_argument("--save_path", type=str, default="../data/world_graphs", help="Path to save the graph plot.")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume the graph from a GML file.")
    parser.add_argument('--max_docs', type=int, default=None, help='Maximum number of documents to process.')
    return parser.parse_args()


def main(data_dir: str, model_base: str, api_key: str, save_path: str, resume: str, max_docs: int) -> None:
    data = load_data(data_dir)
    nb_instances = data_len(data_dir)

    base_model = LiteLLMModel(model_id=model_base, api_key=api_key)
    discovery_manager = AtomicRAGDiscoveryAgentFactory().createAgent(base_model)

    if resume:
        saved_graph = nx.read_gml(resume)
        world_manager = BaseWorldManager(initial_graph=saved_graph)
        discovery_manager.tools['graph_retriever'].update_graph(saved_graph)
        
        nb_seen_instances = len(world_manager.get_worlds())
        nb_instances -= nb_seen_instances
        print(f"Resuming from {resume}. {nb_seen_instances} instances already seen.")
        while nb_seen_instances > 0:
            try:
                next(data)
                nb_seen_instances -= 1
            except StopIteration:
                break
    else:
        world_manager = BaseWorldManager()
        
    try:
        for i, doc in enumerate(tqdm.tqdm(data, total=min(nb_instances, max_docs))):
            if max_docs and i >= max_docs:
                print(f"Processed {max_docs} documents. Stopping.")
                break

            try:
                causal_graph = discovery_manager.run(doc)
                discovery_manager.tools['graph_retriever'].update_graph(causal_graph)
                world_manager.merge(causal_graph)
                
            except Exception as e:
                print(f"Error processing document: {doc}. Error: {e}")

    except KeyboardInterrupt:
        print("Discovery interrupted by user.")
    
    display_plot(world_manager.get_complete_graph(), save_path=save_path)



if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
    