
import os
import sys
import webbrowser
import argparse
import json
import networkx as nx
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for
import threading
import time
import signal
import random
import colorsys


class GraphVisualizer:
    """Main class for the graph visualization tool."""
    
    def __init__(self, port=5000, debug=False, open_browser=True):
        """Initialize the graph visualizer.
        
        Args:
            port (int): Port for the web server
            debug (bool): Whether to run Flask in debug mode
            open_browser (bool): Whether to automatically open the browser
        """
        self.port = port
        self.debug = debug
        self.open_browser = open_browser
        
        # Find the absolute path to the static and templates folders
        self.visualization_dir = os.path.dirname(os.path.abspath(__file__))
        self.static_folder = os.path.join(self.visualization_dir, 'static')
        self.template_folder = os.path.join(self.visualization_dir, 'templates')
        
        # Create Flask app with correct static and template folders
        self.app = Flask(__name__, 
                        static_folder=self.static_folder,
                        template_folder=self.template_folder)
        
        # Setup routes
        self._setup_routes()
        
        # Store the loaded graph
        self.graph = None
        self.graph_path = None

    def _setup_routes(self):
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            """Render the main page."""
            return send_from_directory(self.static_folder, 'index.html')
        
        # Explicitly define routes for CSS and JS files
        @self.app.route('/styles.css')
        def serve_css():
            """Serve the CSS file."""
            return send_from_directory(self.static_folder, 'styles.css')
        
        @self.app.route('/script.js')
        def serve_js():
            """Serve the JavaScript file."""
            return send_from_directory(self.static_folder, 'script.js')
        
        @self.app.route('/api/load-graph', methods=['POST'])
        def load_graph():
            """Load a graph from a file."""
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            
            # Save file temporarily
            temp_path = os.path.join(self.visualization_dir, 'temp_graph')
            file.save(temp_path)
            
            try:
                # Determine the file format based on extension
                _, ext = os.path.splitext(file.filename)
                if ext.lower() == '.gml':
                    graph = nx.read_gml(temp_path)
                elif ext.lower() == '.graphml':
                    graph = nx.read_graphml(temp_path)
                elif ext.lower() == '.gexf':
                    graph = nx.read_gexf(temp_path)
                elif ext.lower() == '.json':
                    with open(temp_path, 'r') as f:
                        graph_data = json.load(f)
                    graph = nx.node_link_graph(graph_data)
                else:
                    # Default to GML
                    graph = nx.read_gml(temp_path)
                
                # Convert the graph to a format suitable for visualization
                self.graph = graph
                self.graph_path = file.filename
                
                # Convert to JSON for the frontend
                graph_data = self._convert_graph_to_json(graph)
                
                return jsonify(graph_data)
            
            except Exception as e:
                return jsonify({'error': f'Error loading graph: {str(e)}'}), 500
            
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        @self.app.route('/api/load-graph-path', methods=['POST'])
        def load_graph_path():
            """Load a graph from a specified path."""
            data = request.json
            if 'path' not in data:
                return jsonify({'error': 'No path provided'}), 400
            
            path = data['path']
            if not os.path.exists(path):
                return jsonify({'error': f'File not found: {path}'}), 404
            
            try:
                # Determine the file format based on extension
                _, ext = os.path.splitext(path)
                if ext.lower() == '.gml':
                    graph = nx.read_gml(path)
                elif ext.lower() == '.graphml':
                    graph = nx.read_graphml(path)
                elif ext.lower() == '.gexf':
                    graph = nx.read_gexf(path)
                elif ext.lower() == '.json':
                    with open(path, 'r') as f:
                        graph_data = json.load(f)
                    graph = nx.node_link_graph(graph_data)
                else:
                    # Default to GML
                    graph = nx.read_gml(path)
                
                # Store the loaded graph
                self.graph = graph
                self.graph_path = path
                
                # Convert to JSON for the frontend
                graph_data = self._convert_graph_to_json(graph)
                
                return jsonify(graph_data)
            
            except Exception as e:
                return jsonify({'error': f'Error loading graph: {str(e)}'}), 500

        @self.app.route('/api/search-world-attribute', methods=['POST'])
        def search_world_attribute():
            """Search for nodes with specific world_i attribute."""
            data = request.json
            if not self.graph:
                return jsonify({'error': 'No graph loaded'}), 400
            
            world_key = data.get('worldKey')
            
            if not world_key:
                return jsonify({'error': 'Missing world key parameter'}), 400
                
            matching_nodes = []
            
            for node, attrs in self.graph.nodes(data=True):
                # Check if this node has the requested world_i attribute
                if world_key in attrs:
                    matching_nodes.append(str(node))
            
            return jsonify({'matchingNodes': matching_nodes})
    
    def _get_node_clusters(self, graph):
        """Detect clusters in the graph using NetworkX community detection.
        
        Args:
            graph (nx.Graph): The NetworkX graph to analyze
            
        Returns:
            dict: A dictionary mapping node IDs to cluster IDs
        """
        try:
            # Try to use community detection algorithms
            from networkx.algorithms import community
            
            # Use Louvain method for community detection if available
            try:
                from community import best_partition
                partition = best_partition(graph)
                return partition
            except ImportError:
                pass
                
            # Fall back to NetworkX's built-in algorithms
            try:
                # Greedy modularity maximization
                communities = list(community.greedy_modularity_communities(graph))
                
                # Convert to a dictionary mapping node to community ID
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
                        
                return partition
            except:
                # If community detection fails, use connected components as clusters
                components = list(nx.connected_components(graph.to_undirected()))
                partition = {}
                for i, comp in enumerate(components):
                    for node in comp:
                        partition[node] = i
                        
                return partition
        except:
            # Fallback: assign all nodes to the same cluster
            return {node: 0 for node in graph.nodes()}
    
    def _generate_distinct_colors(self, num_colors):
        """Generate distinct colors for clusters using HSV color space.
        
        Args:
            num_colors (int): Number of distinct colors to generate
            
        Returns:
            list: List of hex color codes
        """
        colors = []
        for i in range(num_colors):
            # Distribute colors evenly in HSV space
            h = i / num_colors
            s = 0.7 + 0.3 * (i % 3) / 2  # Varying saturation
            v = 0.7 + 0.3 * ((i+1) % 3) / 2  # Varying value/brightness
            
            # Convert HSV to RGB
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            
            # Convert RGB to hex
            color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            colors.append(color)
        
        return colors
    
    def _extract_world_attributes(self, graph):
        """Extract and catalog world_i attributes from the graph.
        
        Args:
            graph (nx.Graph): The NetworkX graph to analyze
            
        Returns:
            list: A list of all world attribute keys found in the graph
        """
        world_keys = set()
        
        for _, attrs in graph.nodes(data=True):
            for key in attrs:
                if key.startswith('world_') and key.split('_')[1].isdigit():
                    world_keys.add(key)
        
        return sorted(list(world_keys))
    
    def _compute_node_positions(self, graph, clusters=None):
        """Compute node positions based on clusters if available.
        
        Args:
            graph (nx.Graph): The NetworkX graph
            clusters (dict): Optional mapping from node to cluster ID
            
        Returns:
            dict: A dictionary mapping node IDs to x,y coordinates
        """
        if clusters:
            # Group nodes by cluster
            cluster_nodes = {}
            for node, cluster in clusters.items():
                if cluster not in cluster_nodes:
                    cluster_nodes[cluster] = []
                cluster_nodes[cluster].append(node)
            
            # Generate a layout for each cluster separately
            positions = {}
            for cluster, nodes in cluster_nodes.items():
                subgraph = graph.subgraph(nodes)
                # Use spring layout for the subgraph
                subgraph_pos = nx.spring_layout(subgraph, seed=42)
                
                # Apply cluster-based positioning
                cluster_center = np.array([cluster % 5 * 500, cluster // 5 * 500])  # Grid layout for clusters
                
                for node, pos in subgraph_pos.items():
                    positions[node] = (pos * 200 + cluster_center).tolist()
            
            return positions
        else:
            # Fallback to standard spring layout
            positions = nx.spring_layout(graph, seed=42)
            return {node: pos.tolist() for node, pos in positions.items()}
    
    def _convert_graph_to_json(self, graph):
        """Convert a networkx graph to a JSON format suitable for visualization.
        
        Args:
            graph (nx.Graph): The networkx graph to convert
            
        Returns:
            dict: Graph data in a format suitable for vis.js
        """
        # Detect clusters in the graph
        clusters = self._get_node_clusters(graph)
        
        # Generate distinct colors for clusters
        num_clusters = max(clusters.values()) + 1 if clusters else 1
        cluster_colors = self._generate_distinct_colors(num_clusters)
        
        # Extract world_i attributes
        world_keys = self._extract_world_attributes(graph)
        
        # Compute initial node positions based on clusters
        positions = self._compute_node_positions(graph, clusters)
        
        # Process nodes with attributes
        nodes = []
        for node, attrs in graph.nodes(data=True):
            # Get cluster ID and color
            cluster_id = clusters.get(node, 0)
            node_color = cluster_colors[cluster_id]
            
            # Get node position
            pos = positions.get(node, [0, 0])
            
            # Check for world_i attributes
            has_world_attrs = False
            for key in attrs:
                if key.startswith('world_') and key.split('_')[1].isdigit():
                    has_world_attrs = True
                    break
            
            node_data = {
                'id': str(node),
                'label': str(node),
                'title': str(node),
                'attributes': {},
                'group': cluster_id,  # Group for coloring
                'hasWorldAttrs': has_world_attrs,
                'x': pos[0],
                'y': pos[1],
                'color': {
                    'background': node_color,
                    'border': '#2c3e50',
                    'highlight': {
                        'background': '#5dade2',
                        'border': '#34495e'
                    }
                },
                'font': {
                    'color': 'white'
                }
            }
            
            # Add all attributes to the node
            for key, value in attrs.items():
                # Convert non-serializable attributes
                if isinstance(value, (int, float, bool, str, list, dict)) or value is None:
                    node_data['attributes'][key] = value
                else:
                    node_data['attributes'][key] = str(value)
            
            nodes.append(node_data)
        
        # Process edges with attributes
        edges = []
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                'from': str(source),
                'to': str(target),
                'attributes': {},
                'arrows': 'to' if graph.is_directed() else '',
                'title': f'Edge: {source} → {target}'  # Tooltip for hover
            }
            
            # Add all attributes to the edge
            for key, value in attrs.items():
                # Convert non-serializable attributes
                if isinstance(value, (int, float, bool, str, list, dict)) or value is None:
                    edge_data['attributes'][key] = value
                else:
                    edge_data['attributes'][key] = str(value)
            
            edges.append(edge_data)
        
        return {
            'nodes': nodes,
            'edges': edges,
            'directed': graph.is_directed(),
            'worldKeys': world_keys,
            'clusters': [{'id': i, 'color': color} for i, color in enumerate(cluster_colors)]
        }
    
    def start(self):
        """Start the visualization server and open browser if requested."""
        # Setup a route to load graph on startup if path is specified
        if self.graph_path and self.graph:
            # If we already have a graph loaded from command-line arguments,
            # add a special route to load it automatically on page load
            @self.app.route('/api/startup-graph')
            def serve_startup_graph():
                """Serve the graph that was loaded at startup."""
                try:
                    # Convert to JSON for the frontend
                    graph_data = self._convert_graph_to_json(self.graph)
                    return jsonify(graph_data)
                except Exception as e:
                    return jsonify({'error': f'Error loading startup graph: {str(e)}'}), 500

        if self.open_browser:
            # Open browser after a short delay to ensure server is up
            url = f'http://localhost:{self.port}'
            if self.graph_path and self.graph:
                # Add a query parameter to indicate that there's a graph to load on startup
                url += '?loadStartupGraph=true'
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        
        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\nShutting down server...")
            # In a production environment, you'd use a proper WSGI server
            # and implement a cleaner shutdown mechanism
            os._exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # Start the Flask app
        self.app.run(host='0.0.0.0', port=self.port, debug=self.debug)


def main():
    """Entry point for the graph visualizer."""
    parser = argparse.ArgumentParser(description='Interactive Graph Visualization Tool')
    parser.add_argument('--port', type=int, default=5000, help='Port for the web server')
    parser.add_argument('--debug', action='store_true', help='Run Flask in debug mode')
    parser.add_argument('--no-browser', dest='open_browser', action='store_false', 
                        help='Do not automatically open browser')
    parser.add_argument('--graph', type=str, help='Path to a graph file to load on startup')
    
    args = parser.parse_args()
    
    visualizer = GraphVisualizer(
        port=args.port,
        debug=args.debug,
        open_browser=args.open_browser
    )
    
    # If a graph file was specified, we'll need to handle it after the server starts
    if args.graph:
        # Validate the file exists
        if not os.path.isfile(args.graph):
            print(f"Error: Graph file not found: {args.graph}")
            return
        
        # We'll need to load this file after the server starts
        # For now we just store it to access later
        visualizer.graph_path = args.graph
    
    print(f"Starting Graph Visualizer on http://localhost:{args.port}")
    print("Press Ctrl+C to stop the server")
    visualizer.start()


if __name__ == '__main__':
    main()



__all__ = ['GraphVisualizer']