import sys
import os
import argparse
from causal_world_modelling_agent.visualization.graph_visualizer import GraphVisualizer

def main():
    """Main entry point for the graph visualizer command."""
    parser = argparse.ArgumentParser(
        description='Interactive visualization tool for causal graphs'
    )
    parser.add_argument(
        '-g', '--graph', 
        help='Path to a graph file to load on startup'
    )
    parser.add_argument(
        '-p', '--port', 
        type=int, 
        default=5000, 
        help='Port for the web server (default: 5000)'
    )
    parser.add_argument(
        '--no-browser', 
        dest='open_browser', 
        action='store_false',
        help='Do not automatically open browser'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Run in debug mode'
    )

    args = parser.parse_args()

    # Validate graph file path if specified
    if args.graph and not os.path.exists(args.graph):
        print(f"Error: Graph file not found: {args.graph}")
        sys.exit(1)

    print(f"Starting Causal Graph Visualizer on http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    # Create a visualizer instance
    visualizer = GraphVisualizer(
        port=args.port,
        debug=args.debug,
        open_browser=args.open_browser
    )
    
    if args.graph:
        # Convert to absolute path if necessary
        abs_graph_path = os.path.abspath(args.graph)
        print(f"Loading graph from: {abs_graph_path}")
        
        # Store the absolute path to load on startup
        visualizer.graph_path = abs_graph_path
        
        # Preload the graph in memory to make sure it loads on startup
        try:
            import networkx as nx
            _, ext = os.path.splitext(abs_graph_path)
            if ext.lower() == '.gml':
                visualizer.graph = nx.read_gml(abs_graph_path)
            elif ext.lower() == '.graphml':
                visualizer.graph = nx.read_graphml(abs_graph_path)
            elif ext.lower() == '.gexf':
                visualizer.graph = nx.read_gexf(abs_graph_path)
            elif ext.lower() == '.json':
                import json
                with open(abs_graph_path, 'r') as f:
                    graph_data = json.load(f)
                visualizer.graph = nx.node_link_graph(graph_data)
            else:
                print(f"Warning: Unrecognized file extension {ext}. Attempting to read as GML.")
                visualizer.graph = nx.read_gml(abs_graph_path)
            
            print(f"Successfully loaded graph with {visualizer.graph.number_of_nodes()} nodes and {visualizer.graph.number_of_edges()} edges")
        except Exception as e:
            print(f"Error preloading graph: {str(e)}")
            print("Will attempt to load again when server starts.")
    
    visualizer.start()

if __name__ == "__main__":
    main()