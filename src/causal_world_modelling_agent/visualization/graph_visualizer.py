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
from collections import defaultdict, Counter
import warnings
import re
import tempfile
from typing import Dict, Any, List, Optional, Union

# Import for the improved semantic topic clustering
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    SEMANTIC_CLUSTERING_AVAILABLE = True
except ImportError:
    SEMANTIC_CLUSTERING_AVAILABLE = False
    warnings.warn("sentence-transformers and/or scikit-learn not found. Topic clustering will fall back to simple word frequency analysis.")

# Try to import community package (python-louvain)
try:
    import community as community_louvain
    COMMUNITY_PACKAGE_AVAILABLE = True
except ImportError:
    COMMUNITY_PACKAGE_AVAILABLE = False
    warnings.warn("python-louvain package not found. Community detection will use NetworkX's algorithms instead.")

# Use NetworkX's community detection algorithms as fallback
from networkx.algorithms import community

# Custom JSON encoder to handle NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

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
        
        # Configure app to use the custom JSON encoder
        self.app.json_encoder = NumpyJSONEncoder
        
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
            
        @self.app.route('/api/analyze-graph', methods=['POST'])
        def analyze_graph():
            """Analyze the currently loaded graph and return comprehensive metrics."""
            if not self.graph:
                return jsonify({'error': 'No graph loaded'}), 400
                
            data = request.json
            analysis_type = data.get('type', 'all')
            
            try:
                result = None
                
                if analysis_type == 'structural':
                    result = self.analyze_structural_properties()
                elif analysis_type == 'causal':
                    result = self.analyze_causal_properties()
                elif analysis_type == 'community':
                    result = self.analyze_community_structure()
                elif analysis_type == 'topic':
                    result = self.analyze_topic_clusters()
                else:
                    # Perform all analyses
                    result = {
                        'structural': self.analyze_structural_properties(),
                        'causal': self.analyze_causal_properties(),
                        'community': self.analyze_community_structure(),
                        'topic': self.analyze_topic_clusters()
                    }
                
                # Add debug information to the result
                if self.debug:
                    result['debug_info'] = {
                        'graph_nodes': self.graph.number_of_nodes(),
                        'graph_edges': self.graph.number_of_edges(),
                        'is_directed': self.graph.is_directed(),
                        'analysis_type': analysis_type,
                        'graph_path': self.graph_path
                    }
                
                return jsonify(result)
                
            except Exception as e:
                import traceback
                error_details = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'traceback': traceback.format_exc()
                }
                
                # Add graph information for debugging
                if self.debug:
                    error_details['debug_info'] = {
                        'graph_nodes': self.graph.number_of_nodes() if self.graph else 0,
                        'graph_edges': self.graph.number_of_edges() if self.graph else 0,
                        'is_directed': self.graph.is_directed() if self.graph else None,
                        'analysis_type': analysis_type,
                        'graph_path': self.graph_path
                    }
                
                # Print detailed error to server console
                print(f"Error in analyze_graph ({analysis_type}):")
                print(traceback.format_exc())
                
                return jsonify(error_details), 500
    
    def analyze_structural_properties(self):
        """Analyze streamlined structural properties of the graph.
        
        Returns:
            dict: Dictionary containing structural metrics
        """
        if not self.graph:
            raise ValueError("No graph loaded")
        
        graph = self.graph
        
        # Density
        density = nx.density(graph)
        
        # Degree statistics
        in_degree = dict(graph.in_degree()) if graph.is_directed() else {}
        out_degree = dict(graph.out_degree()) if graph.is_directed() else {}
        
        # Get degree distribution
        in_degree_dist = Counter(in_degree.values()) if in_degree else Counter()
        out_degree_dist = Counter(out_degree.values()) if out_degree else Counter()
        
        # Format degree distributions for visualization
        in_degree_dist_data = {
            'labels': [str(k) for k in sorted(in_degree_dist.keys())],
            'values': [in_degree_dist[k] for k in sorted(in_degree_dist.keys())]
        }
        
        out_degree_dist_data = {
            'labels': [str(k) for k in sorted(out_degree_dist.keys())],
            'values': [out_degree_dist[k] for k in sorted(out_degree_dist.keys())]
        }
        
        # Component analysis
        if graph.is_directed():
            strongly_connected_components = list(nx.strongly_connected_components(graph))
            weakly_connected_components = list(nx.weakly_connected_components(graph))
            
            component_counts = {
                'strongly_connected': len(strongly_connected_components),
                'weakly_connected': len(weakly_connected_components),
            }
        else:
            connected_components = list(nx.connected_components(graph))
            component_counts = {
                'connected': len(connected_components),
            }
            
        # Path metrics for the largest component
        path_metrics = {}
        if graph.is_directed() and nx.is_weakly_connected(graph):
            largest_wcc = max(nx.weakly_connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_wcc).copy()
            try:
                path_metrics['diameter'] = nx.diameter(subgraph.to_undirected())
                path_metrics['average_path_length'] = nx.average_shortest_path_length(subgraph.to_undirected())
            except:
                path_metrics['diameter'] = "N/A - Graph not strongly connected"
                path_metrics['average_path_length'] = "N/A - Graph not strongly connected"
        elif not graph.is_directed() and nx.is_connected(graph):
            try:
                path_metrics['diameter'] = nx.diameter(graph)
                path_metrics['average_path_length'] = nx.average_shortest_path_length(graph)
            except:
                path_metrics['diameter'] = "N/A"
                path_metrics['average_path_length'] = "N/A"
        else:
            largest_cc = max(nx.connected_components(graph.to_undirected()) 
                          if graph.is_directed() else nx.connected_components(graph), key=len)
            subgraph = graph.subgraph(largest_cc).copy()
            try:
                path_metrics['diameter'] = nx.diameter(subgraph.to_undirected() if graph.is_directed() else subgraph)
                path_metrics['average_path_length'] = nx.average_shortest_path_length(
                    subgraph.to_undirected() if graph.is_directed() else subgraph
                )
                path_metrics['note'] = "Metrics computed on largest connected component"
            except:
                path_metrics['diameter'] = "N/A"
                path_metrics['average_path_length'] = "N/A"
                
        return {
            'density': density,
            'in_degree_distribution': in_degree_dist_data,
            'out_degree_distribution': out_degree_dist_data,
            'components': component_counts,
            'path_metrics': path_metrics
        }
        
    def analyze_causal_properties(self):
        """Analyze streamlined causal-specific properties of the graph.
        
        Returns:
            dict: Dictionary containing causal metrics
        """
        if not self.graph:
            raise ValueError("No graph loaded")
            
        graph = self.graph
        is_directed = graph.is_directed()
        
        if not is_directed:
            return {
                'error': 'Causal analysis requires a directed graph',
                'is_directed': False
            }
        
        root_causes = [str(node) for node, in_deg in graph.in_degree() if in_deg == 0]
        terminal_effects = [str(node) for node, out_deg in graph.out_degree() if out_deg == 0]
        
        longest_chain = []
        for source in root_causes:
            for target in terminal_effects:
                try:
                    paths = list(nx.all_simple_paths(graph, source, target))
                    if paths and len(paths[0]) > len(longest_chain):
                        longest_chain = paths[0]
                except:
                    continue
                    
        longest_chain = [str(node) for node in longest_chain]
        
        betweenness = nx.betweenness_centrality(graph)
        sorted_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
        bottlenecks = [{"node": str(node), "betweenness": value} 
                      for node, value in sorted_betweenness[:min(10, len(sorted_betweenness))]]
        
        intervention_impact = {}
        for node in graph.nodes():
            try:
                reachable = nx.descendants(graph, node)
                intervention_impact[str(node)] = len(reachable)
            except:
                intervention_impact[str(node)] = 0
                
        sorted_impact = sorted(intervention_impact.items(), key=lambda x: x[1], reverse=True)
        top_intervention_nodes = [{"node": node, "impact": impact} 
                                for node, impact in sorted_impact[:min(10, len(sorted_impact))]]
                              
        feedback_loop_count = 0
        try:
            cycles = list(nx.simple_cycles(graph))
            feedback_loop_count = len([cycle for cycle in cycles if len(cycle) > 1])
        except:
            pass
        
        return {
            'root_causes_count': len(root_causes),
            'terminal_effects_count': len(terminal_effects),
            'longest_chain': longest_chain if longest_chain else None,
            'longest_chain_length': len(longest_chain) if longest_chain else 0,
            'bottlenecks': bottlenecks,
            'top_intervention_nodes': top_intervention_nodes,
            'feedback_loops_count': feedback_loop_count
        }
        
    def analyze_community_structure(self):
        """Analyze streamlined community structure of the graph.
        
        Returns:
            dict: Dictionary containing community metrics
        """
        if not self.graph:
            raise ValueError("No graph loaded")
            
        graph = self.graph
        undirected_graph = graph.to_undirected()
        
        community_info = {}
        
        # Try to detect communities
        partition = None
        
        try:
            # Try Louvain method if available
            if COMMUNITY_PACKAGE_AVAILABLE:
                try:
                    partition = community_louvain.best_partition(undirected_graph)
                    community_info['method'] = 'Louvain'
                except Exception as e:
                    community_info['louvain_error'] = str(e)
                    partition = None
            
            # Fallback to NetworkX's algorithms if Louvain failed or is unavailable
            if partition is None:
                try:
                    # Use Greedy Modularity Maximization from NetworkX
                    communities = list(community.greedy_modularity_communities(undirected_graph))
                    
                    # Convert to dictionary format like Louvain
                    partition = {}
                    for comm_id, comm_nodes in enumerate(communities):
                        for node in comm_nodes:
                            partition[node] = comm_id
                    
                    community_info['method'] = 'NetworkX Greedy Modularity'
                except Exception as e:
                    community_info['nx_community_error'] = str(e)
                    
                    # Last resort: use connected components as "communities"
                    try:
                        components = list(nx.connected_components(undirected_graph))
                        partition = {}
                        for comm_id, comp in enumerate(components):
                            for node in comp:
                                partition[node] = comm_id
                        community_info['method'] = 'Connected Components'
                    except Exception as e:
                        community_info['error'] = f"Failed to detect communities: {str(e)}"
                        return community_info
            
            # Calculate community count
            communities = set(partition.values())
            community_info['count'] = len(communities)
            
            # Find community bridging nodes
            bridging_nodes = []
            
            # For each node, check if its neighbors belong to different communities
            for node in graph.nodes():
                node_comm = partition.get(node)
                if node_comm is not None:
                    neighbor_comms = set()
                    for neighbor in graph.neighbors(node):
                        neighbor_comm = partition.get(neighbor)
                        if neighbor_comm is not None and neighbor_comm != node_comm:
                            neighbor_comms.add(neighbor_comm)
                    
                    if neighbor_comms:
                        bridging_nodes.append({
                            'node': str(node),
                            'community': node_comm,
                            'bridges_to': list(neighbor_comms),
                            'bridge_count': len(neighbor_comms)
                        })
            
            # Sort by number of communities bridged
            bridging_nodes.sort(key=lambda x: x['bridge_count'], reverse=True)
            community_info['bridging_nodes'] = bridging_nodes[:min(50, len(bridging_nodes))]
        
        except Exception as e:
            community_info['error'] = str(e)
        
        return community_info
        
    def analyze_topic_clusters(self):
        """Cluster nodes based on topics from text attributes using semantic embeddings.
        
        Returns:
            dict: Dictionary containing topic clusters
        """
        if not self.graph:
            raise ValueError("No graph loaded")
            
        graph = self.graph
        
        # Step 1: Extract text from node attributes
        node_text = {}
        
        for node, attrs in graph.nodes(data=True):
            node_str = str(node)
            text_content = ""
            # Attributes likely to contain descriptive text
            text_attributes = ['name', 'label', 'description', 'contextual_information', 
                               'supporting_text', 'title', 'content', 'text']
            
            for attr in text_attributes:
                if attr in attrs and attrs[attr]:
                    content = str(attrs[attr])
                    text_content += " " + content
            
            # Also include any world attributes which often contain semantic information
            for key, value in attrs.items():
                if isinstance(value, str):
                    if key.startswith('world_') or any(term in key for term in ['desc', 'text', 'info']):
                        text_content += " " + value
            
            # Only include nodes with actual text content
            if text_content.strip():
                # Clean up text - remove special characters and excessive whitespace
                text_content = re.sub(r'[^\w\s]', ' ', text_content)
                text_content = re.sub(r'\s+', ' ', text_content).strip()
                node_text[node_str] = text_content
        
        # If we have fewer than 3 nodes with text, fall back to simple clustering
        if len(node_text) < 3:
            return self._fallback_topic_clustering(node_text)
        
        # Use semantic clustering if available, otherwise fall back
        if SEMANTIC_CLUSTERING_AVAILABLE:
            return self._semantic_topic_clustering(node_text)
        else:
            return self._fallback_topic_clustering(node_text)
    
    def _semantic_topic_clustering(self, node_text):
        """Use sentence transformers to create semantic clusters.
        
        Args:
            node_text (dict): Dictionary mapping node IDs to text content
            
        Returns:
            dict: Dictionary containing topic clusters
        """
        try:
            # Load a pre-trained sentence transformer model
            # Using all-mpnet-base-v2 for high quality sentence embeddings
            model = SentenceTransformer('all-mpnet-base-v2')
            
            # Create node IDs and texts lists to maintain consistent order
            node_ids = list(node_text.keys())
            texts = [node_text[node_id] for node_id in node_ids]
            
            # Generate embeddings for all texts
            embeddings = model.encode(texts, show_progress_bar=False)
            
            # Ensure embeddings are regular Python lists, not NumPy arrays
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            # Determine the optimal number of clusters using silhouette score
            # Try different cluster counts to find best separation
            max_clusters = min(20, len(node_text) - 1)  # Don't try more clusters than we have nodes - 1
            best_score = -1
            best_k = 2  # Default to 2 clusters if we can't find better
            
            if len(node_text) > 10:  # Only worth trying multiple k values if we have enough data
                for k in range(2, min(max_clusters, 10) + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    # Convert NumPy array to Python list if needed
                    if isinstance(cluster_labels, np.ndarray):
                        cluster_labels = cluster_labels.tolist()
                    
                    # Skip if we have any empty clusters
                    if len(set(cluster_labels)) < k:
                        continue
                    
                    # Calculate silhouette score
                    if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                        score = silhouette_score(embeddings, cluster_labels)
                        # Convert NumPy float to Python float
                        if isinstance(score, np.floating):
                            score = float(score)
                        if score > best_score:
                            best_score = score
                            best_k = k
            
            # Apply K-means clustering with the optimal number of clusters
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Convert NumPy array to Python list if needed
            if isinstance(cluster_labels, np.ndarray):
                cluster_labels = cluster_labels.tolist()
            
            # Group nodes by cluster
            clusters = defaultdict(list)
            for i, node_id in enumerate(node_ids):
                clusters[int(cluster_labels[i])].append(node_id)
            
            # Extract representative terms for each cluster using TF-IDF like approach
            cluster_topics = {}
            for cluster_id, nodes in clusters.items():
                # Get all texts in this cluster
                cluster_texts = [node_text[node] for node in nodes]
                combined_text = " ".join(cluster_texts).lower()
                
                # Extract word frequencies but exclude stopwords and short words
                words = combined_text.split()
                word_counts = Counter(word for word in words 
                                    if word.lower() not in ENGLISH_STOP_WORDS 
                                    and len(word) > 3)
                
                # Calculate a simple tf-idf inspired score
                # Count how many texts each word appears in
                word_doc_frequency = {}
                for word in word_counts:
                    word_doc_frequency[word] = sum(1 for text in cluster_texts if word in text.lower())
                
                # Calculate importance score: frequency * (texts_containing / total_texts)
                # This prioritizes words that appear frequently but are concentrated in this cluster
                word_scores = {}
                for word, count in word_counts.items():
                    # Term frequency × document frequency within cluster / total documents
                    word_scores[word] = count * (word_doc_frequency[word] / len(cluster_texts))
                
                # Get top words
                top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                
                if top_words:
                    # Use the most significant word as the cluster topic
                    cluster_topics[cluster_id] = top_words[0][0]
                else:
                    cluster_topics[cluster_id] = f"cluster_{cluster_id}"
            
            # Format the clusters for frontend display
            topic_clusters = []
            for cluster_id, nodes in clusters.items():
                topic = cluster_topics[cluster_id]
                topic_clusters.append({
                    "topic": topic,
                    "nodes": nodes,
                    "size": len(nodes),
                    "cluster_id": int(cluster_id)  # Ensure cluster_id is a Python int, not NumPy int
                })
            
            # Sort clusters by size
            topic_clusters.sort(key=lambda x: x["size"], reverse=True)
            
            # Ensure that any NumPy values are converted to Python native types
            best_score = float(best_score) if best_score > -1 else None
            
            # Return the result
            return {
                'topic_clusters': topic_clusters,
                'cluster_count': len(topic_clusters),
                'nodes_with_topics': len(node_text),
                'nodes_without_topics': len(self.graph.nodes()) - len(node_text),
                'method': 'semantic_embeddings',
                'model': 'all-mpnet-base-v2',
                'silhouette_score': best_score
            }
        
        except Exception as e:
            # If anything fails with the semantic approach, fall back to simple clustering
            print(f"Semantic clustering failed: {str(e)}")
            return self._fallback_topic_clustering(node_text)
    
    def _fallback_topic_clustering(self, node_text):
        """Fall back to simple word frequency-based topic clustering.
        
        Args:
            node_text (dict): Dictionary mapping node IDs to text content
            
        Returns:
            dict: Dictionary containing topic clusters
        """
        node_topics = {}
        
        for node, text in node_text.items():
            if text:
                stopwords = set(['the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'was', 'for', 
                               'on', 'with', 'by', 'as', 'it', 'from', 'at', 'an', 'be', 'this',
                               'are', 'or', 'not', 'which'])
                
                words = text.lower().split()
                word_counts = Counter(word for word in words 
                                   if word not in stopwords and len(word) > 3)
                                   
                node_topics[node] = [
                    {"word": word, "count": count}
                    for word, count in word_counts.most_common(5)
                ] if word_counts else []
        
        topic_clusters = defaultdict(list)
        
        for node, topics in node_topics.items():
            if topics:
                main_topic = topics[0]["word"]
                topic_clusters[main_topic].append(node)
                
        sorted_topic_clusters = sorted(
            [{"topic": topic, "nodes": nodes, "size": len(nodes)} 
             for topic, nodes in topic_clusters.items()],
            key=lambda x: x["size"],
            reverse=True
        )
        
        return {
            'topic_clusters': sorted_topic_clusters[:20],
            'cluster_count': len(topic_clusters),
            'nodes_with_topics': len(node_topics),
            'nodes_without_topics': len(self.graph.nodes()) - len(node_topics),
            'method': 'word_frequency'
        }
    
    def _get_node_clusters(self, graph):
        """Detect clusters in the graph using community detection.
        
        Args:
            graph (nx.Graph): The NetworkX graph to analyze
            
        Returns:
            dict: A dictionary mapping node IDs to cluster IDs
        """
        try:
            # Convert to undirected graph for community detection
            undirected_graph = graph.to_undirected()
            
            # Try Louvain method if available
            if COMMUNITY_PACKAGE_AVAILABLE:
                try:
                    partition = community_louvain.best_partition(undirected_graph)
                    return partition
                except Exception:
                    pass  # Fall through to next method if this fails
            
            # Fall back to NetworkX's built-in algorithms
            try:
                # Greedy modularity maximization
                communities = list(community.greedy_modularity_communities(undirected_graph))
                
                # Convert to a dictionary mapping node to community ID
                partition = {}
                for i, comm in enumerate(communities):
                    for node in comm:
                        partition[node] = i
                        
                return partition
            except Exception:
                # If community detection fails, use connected components as clusters
                components = list(nx.connected_components(undirected_graph))
                partition = {}
                for i, comp in enumerate(components):
                    for node in comp:
                        partition[node] = i
                        
                return partition
        except Exception:
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
            cluster_nodes = {}
            for node, cluster in clusters.items():
                if cluster not in cluster_nodes:
                    cluster_nodes[cluster] = []
                cluster_nodes[cluster].append(node)
            
            positions = {}
            for cluster, nodes in cluster_nodes.items():
                subgraph = graph.subgraph(nodes)
                subgraph_pos = nx.spring_layout(subgraph, seed=42)
                cluster_center = np.array([cluster % 5 * 500, cluster // 5 * 500])
                
                for node, pos in subgraph_pos.items():
                    positions[node] = (pos * 200 + cluster_center).tolist()
            
            return positions
        else:
            positions = nx.spring_layout(graph, seed=42)
            return {node: pos.tolist() for node, pos in positions.items()}
    
    def _convert_graph_to_json(self, graph):
        """Convert a networkx graph to a JSON format suitable for visualization.
        
        Args:
            graph (nx.Graph): The networkx graph to convert
            
        Returns:
            dict: Graph data in a format suitable for vis.js
        """
        clusters = self._get_node_clusters(graph)
        
        num_clusters = max(clusters.values()) + 1 if clusters else 1
        cluster_colors = self._generate_distinct_colors(num_clusters)
        
        world_keys = self._extract_world_attributes(graph)
        
        positions = self._compute_node_positions(graph, clusters)
        
        nodes = []
        for node, attrs in graph.nodes(data=True):
            cluster_id = clusters.get(node, 0)
            node_color = cluster_colors[cluster_id]
            
            pos = positions.get(node, [0, 0])
            
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
                'group': cluster_id,
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
            
            for key, value in attrs.items():
                if isinstance(value, (int, float, bool, str, list, dict)) or value is None:
                    node_data['attributes'][key] = value
                else:
                    node_data['attributes'][key] = str(value)
            
            nodes.append(node_data)
        
        edges = []
        for source, target, attrs in graph.edges(data=True):
            edge_data = {
                'from': str(source),
                'to': str(target),
                'attributes': {},
                'arrows': 'to' if graph.is_directed() else '',
                'title': f'Edge: {source} → {target}'
            }
            
            for key, value in attrs.items():
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
        if self.graph_path and self.graph:
            @self.app.route('/api/startup-graph')
            def serve_startup_graph():
                """Serve the graph that was loaded at startup."""
                try:
                    graph_data = self._convert_graph_to_json(self.graph)
                    return jsonify(graph_data)
                except Exception as e:
                    return jsonify({'error': f'Error loading startup graph: {str(e)}'}), 500

        if self.open_browser:
            url = f'http://localhost:{self.port}'
            if self.graph_path and self.graph:
                url += '?loadStartupGraph=true'
            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        
        def signal_handler(sig, frame):
            print("\nShutting down server...")
            os._exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        self.app.run(host='0.0.0.0', port=self.port, debug=self.debug)


__all__ = ['GraphVisualizer']