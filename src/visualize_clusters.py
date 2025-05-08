import os
import sys
import argparse
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection  # Import LineCollection from correct module
import re
from collections import defaultdict, Counter
import warnings
import json
import colorsys
from pathlib import Path

# Import for dimensionality reduction and clustering
try:
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    from sentence_transformers import SentenceTransformer
    ADVANCED_CLUSTERING_AVAILABLE = True
except ImportError:
    ADVANCED_CLUSTERING_AVAILABLE = False
    warnings.warn("scikit-learn, sentence-transformers or related packages not found. "
                 "Install them with: pip install scikit-learn sentence-transformers matplotlib")

# Try to import community package (python-louvain)
try:
    import community as community_louvain
    COMMUNITY_PACKAGE_AVAILABLE = True
except ImportError:
    COMMUNITY_PACKAGE_AVAILABLE = False
    warnings.warn("python-louvain package not found. Some community detection methods will be unavailable.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize node clusters in a graph using T-SNE and various clustering algorithms"
    )
    parser.add_argument(
        "-g", "--graph", required=True,
        help="Path to the graph file (.gml, .graphml, .gexf, or .json)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output file path (defaults to clusters_[TIMESTAMP].png)"
    )
    parser.add_argument(
        "--format", choices=["png", "pdf", "svg"], default="png",
        help="Output file format (default: png)"
    )
    parser.add_argument(
        "--algorithm", choices=["louvain", "kmeans", "dbscan", "agglomerative", "structural"], 
        default="kmeans",
        help="Clustering algorithm to use (default: kmeans)"
    )
    parser.add_argument(
        "--clusters", type=int, default=0,
        help="Number of clusters to create (default: auto-detect optimal number)"
    )
    parser.add_argument(
        "--perplexity", type=float, default=30.0,
        help="Perplexity parameter for T-SNE (default: 30.0)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=200.0,
        help="Learning rate for T-SNE (default: 200.0)"
    )
    parser.add_argument(
        "--no-text", action="store_true",
        help="Disable text extraction and use only graph structure for clustering"
    )
    parser.add_argument(
        "--figsize", default="16,12",
        help="Figure size in inches as width,height (default: 16,12)"
    )
    parser.add_argument(
        "--title", type=str, default="",
        help="Custom title for the visualization"
    )
    parser.add_argument(
        "--dpi", type=int, default=100,
        help="DPI for the output image (default: 100)"
    )
    parser.add_argument(
        "--iterations", type=int, default=1000,
        help="Number of iterations for T-SNE (default: 1000)"
    )
    parser.add_argument(
        "--topic-method", choices=["frequency", "ngram", "representative", "embedding"], 
        default="frequency",
        help="Method for extracting cluster topics (default: frequency)"
    )
    parser.add_argument(
        "--cluster-on-reduced-space", action="store_true",
        help="Cluster on reduced space instead of original embeddings"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    return parser.parse_args()


def load_graph(graph_path):
    """Load graph from file based on extension."""
    print(f"Loading graph from {graph_path}...")
    try:
        _, ext = os.path.splitext(graph_path)
        if ext.lower() == '.gml':
            G = nx.read_gml(graph_path)
        elif ext.lower() == '.graphml':
            G = nx.read_graphml(graph_path)
        elif ext.lower() == '.gexf':
            G = nx.read_gexf(graph_path)
        elif ext.lower() == '.json':
            import json
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
            G = nx.node_link_graph(graph_data)
        else:
            print(f"Warning: Unrecognized file extension {ext}. Attempting to read as GML.")
            G = nx.read_gml(graph_path)
        
        print(f"Successfully loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"Error loading graph: {str(e)}")
        raise


def extract_node_text(graph):
    """Extract text from node attributes for semantic clustering."""
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
    
    return node_text


def generate_embeddings(node_text):
    """Generate semantic embeddings using sentence transformers."""
    if not ADVANCED_CLUSTERING_AVAILABLE:
        print("Advanced clustering libraries not available. Please install required packages.")
        sys.exit(1)
        
    print("Generating semantic embeddings...")
    model = SentenceTransformer('all-mpnet-base-v2')
    
    # Create node IDs and texts lists to maintain consistent order
    node_ids = list(node_text.keys())
    texts = [node_text[node_id] for node_id in node_ids]
    
    # Generate embeddings for all texts
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return node_ids, embeddings


def compute_structural_features(graph):
    """Compute structural features for nodes when text is not available."""
    print("Computing structural features for nodes...")
    
    # Get basic node features
    features = {}
    
    # Convert to undirected for some metrics
    undirected = graph.to_undirected()
    
    # Compute degree centrality
    in_degree = dict(graph.in_degree()) if graph.is_directed() else {}
    out_degree = dict(graph.out_degree()) if graph.is_directed() else {}
    degree = dict(undirected.degree())
    
    # Compute other centrality measures
    try:
        betweenness = nx.betweenness_centrality(graph)
    except:
        betweenness = {node: 0 for node in graph.nodes()}
    
    try:
        closeness = nx.closeness_centrality(graph)
    except:
        closeness = {node: 0 for node in graph.nodes()}
    
    try:
        clustering = nx.clustering(undirected)
    except:
        clustering = {node: 0 for node in graph.nodes()}
    
    # Combine features
    for node in graph.nodes():
        node_str = str(node)
        node_features = [
            in_degree.get(node, 0) if graph.is_directed() else 0,
            out_degree.get(node, 0) if graph.is_directed() else 0,
            degree.get(node, 0),
            betweenness.get(node, 0),
            closeness.get(node, 0),
            clustering.get(node, 0)
        ]
        features[node_str] = node_features
    
    # Convert to list format
    node_ids = list(features.keys())
    feature_matrix = np.array([features[node_id] for node_id in node_ids])
    
    # Standardize features
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    return node_ids, feature_matrix


def reduce_dimensions(features, perplexity=30, learning_rate=200, iterations=1000, seed=42):
    """Reduce dimensions using T-SNE."""
    print(f"Reducing dimensions with T-SNE (perplexity={perplexity}, lr={learning_rate}, iterations={iterations})...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=iterations,
        random_state=seed,
        verbose=1
    )
    return tsne.fit_transform(features)


def cluster_nodes(embeddings, algorithm="kmeans", n_clusters=0, seed=42):
    """Cluster nodes using the specified algorithm."""
    print(f"Clustering nodes using {algorithm} algorithm...")
    
    # Determine optimal number of clusters if not specified
    if n_clusters <= 0:
        max_clusters = min(50, embeddings.shape[0] - 1)
        best_score = -1
        best_k = 2
        
        if embeddings.shape[0] > 10:
            for k in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10)
                cluster_labels = kmeans.fit_predict(embeddings)
                
                if len(set(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                    try:
                        score = silhouette_score(embeddings, cluster_labels)
                        print(f"  k={k}, silhouette score={score:.4f}")
                        if score > best_score:
                            best_score = score
                            best_k = k
                    except:
                        continue
        n_clusters = best_k
    
    print(f"Using {n_clusters} clusters")
    
    # Apply the selected clustering algorithm
    if algorithm == "kmeans":
        clustering = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return clustering.fit_predict(embeddings)
    elif algorithm == "dbscan":
        # DBSCAN automatically determines the number of clusters
        eps = 0.5  # Radius parameter - distance threshold
        min_samples = max(5, embeddings.shape[0] // 100)  # Minimum points in neighborhood
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
        return clustering.fit_predict(embeddings)
    elif algorithm == "agglomerative":
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        return clustering.fit_predict(embeddings)
    elif algorithm == "louvain":
        if not COMMUNITY_PACKAGE_AVAILABLE:
            print("Louvain algorithm not available, falling back to KMeans.")
            clustering = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
            return clustering.fit_predict(embeddings)
        else:
            # Create temporary graph for Louvain
            temp_graph = nx.Graph()
            for i in range(embeddings.shape[0]):
                temp_graph.add_node(i)
            
            # Connect similar nodes based on their embeddings
            for i in range(embeddings.shape[0]):
                for j in range(i+1, embeddings.shape[0]):
                    # Calculate similarity using Euclidean distance
                    distance = np.linalg.norm(embeddings[i] - embeddings[j])
                    if distance < 0.5:  # Threshold for connection
                        temp_graph.add_edge(i, j, weight=1.0-distance)
            
            # Apply Louvain community detection
            partition = community_louvain.best_partition(temp_graph)
            return np.array([partition[i] for i in range(embeddings.shape[0])])
    elif algorithm == "structural":
        # This would be using structural features, not embeddings
        # But the function interface is the same
        clustering = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return clustering.fit_predict(embeddings)
    else:
        print(f"Unknown algorithm: {algorithm}, falling back to KMeans")
        clustering = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return clustering.fit_predict(embeddings)


def extract_cluster_topics(node_text, node_ids, labels):
    """Extract representative topics for each cluster based on text content."""
    print("Extracting representative topics for each cluster...")
    
    cluster_texts = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        if i < len(node_ids) and node_ids[i] in node_text:
            cluster_texts[cluster_id].append(node_text[node_ids[i]])
    
    topics = {}
    for cluster_id, texts in cluster_texts.items():
        if not texts:
            topics[cluster_id] = f"Cluster {cluster_id}"
            continue
            
        # Combine texts in this cluster
        combined_text = " ".join(texts).lower()
        
        # Extract word frequencies but exclude stopwords and short words
        words = combined_text.split()
        word_counts = Counter(word for word in words 
                             if word.lower() not in ENGLISH_STOP_WORDS 
                             and len(word) > 3)
        
        # Calculate word importance
        word_doc_frequency = {}
        for word in word_counts:
            word_doc_frequency[word] = sum(1 for text in texts if word in text.lower())
        
        word_scores = {}
        for word, count in word_counts.items():
            word_scores[word] = count * (word_doc_frequency[word] / len(texts))
        
        # Get top words
        top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        
        if top_words:
            topics[cluster_id] = top_words[0][0].title()
        else:
            topics[cluster_id] = f"Cluster {cluster_id}"
    
    return topics


def extract_ngram_topics(node_text, node_ids, labels):
    """Extract representative topics for each cluster using bigrams and trigrams."""
    print("Extracting n-gram topics for each cluster...")
    
    # Group text by cluster
    cluster_texts = defaultdict(list)
    for i, cluster_id in enumerate(labels):
        if i < len(node_ids) and node_ids[i] in node_text:
            cluster_texts[cluster_id].append(node_text[node_ids[i]])
    
    topics = {}
    for cluster_id, texts in cluster_texts.items():
        if not texts:
            topics[cluster_id] = f"Cluster {cluster_id}"
            continue
            
        # Combine texts in this cluster
        combined_text = " ".join(texts).lower()
        
        # Clean text
        combined_text = re.sub(r'[^\w\s]', ' ', combined_text)
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()
        words = combined_text.split()
        
        # Generate n-grams
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1) 
                  if words[i].lower() not in ENGLISH_STOP_WORDS 
                  and words[i+1].lower() not in ENGLISH_STOP_WORDS
                  and len(words[i]) > 2 and len(words[i+1]) > 2]
        
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2) 
                   if words[i].lower() not in ENGLISH_STOP_WORDS 
                   and words[i+2].lower() not in ENGLISH_STOP_WORDS
                   and len(words[i]) > 2 and len(words[i+2]) > 2]
        
        # Count frequencies
        bigram_counts = Counter(bigrams)
        trigram_counts = Counter(trigrams)
        word_counts = Counter([w for w in words if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 3])
        
        # Calculate n-gram significance
        bigram_scores = {bigram: count * 1.2 for bigram, count in bigram_counts.most_common(10)}
        trigram_scores = {trigram: count * 1.5 for trigram, count in trigram_counts.most_common(5)}
        word_scores = {word: count for word, count in word_counts.most_common(20)}
        
        # Combine all scores
        all_scores = {}
        all_scores.update(word_scores)
        all_scores.update(bigram_scores)
        all_scores.update(trigram_scores)
        
        # Get top phrases
        top_phrases = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        
        if top_phrases:
            # Capitalize each word in the phrase
            topic_words = top_phrases[0][0].split()
            topics[cluster_id] = ' '.join([word.title() for word in topic_words])
        else:
            topics[cluster_id] = f"Cluster {cluster_id}"
    
    return topics


def extract_representative_topics(graph, node_ids, labels):
    """Extract topics by finding the most representative node (highest centrality) in each cluster."""
    print("Extracting representative node topics for each cluster...")
    
    # Convert to undirected for centrality calculation
    undirected = graph.to_undirected()
    
    # Calculate node centrality
    try:
        # Eigenvector centrality gives importance based on connections to important nodes
        centrality = nx.eigenvector_centrality_numpy(undirected)
    except:
        # Fall back to degree centrality if eigenvector fails
        centrality = nx.degree_centrality(undirected)
    
    # Group nodes by cluster
    cluster_nodes = defaultdict(list)
    
    for i, node in enumerate(graph.nodes()):
        if i < len(labels):
            cluster_id = labels[i]
            # Store node and its centrality score
            cluster_nodes[cluster_id].append((node, centrality.get(node, 0)))
    
    # Find most central node in each cluster
    topics = {}
    for cluster_id, nodes in cluster_nodes.items():
        if not nodes:
            topics[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        # Sort by centrality (descending)
        sorted_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)
        central_node = sorted_nodes[0][0]
        
        # Get node name or title
        node_attrs = graph.nodes[central_node]
        
        # Look for descriptive attributes in order of preference
        for attr in ['name', 'title', 'label', 'id']:
            if attr in node_attrs and node_attrs[attr]:
                # Clean and format the name
                name = str(node_attrs[attr])
                # Replace underscores with spaces and capitalize words
                name = name.replace('_', ' ')
                name = ' '.join(word.capitalize() for word in name.split())
                topics[cluster_id] = name
                break
        else:
            # If no good name found, use the node's string representation
            topics[cluster_id] = str(central_node).replace('_', ' ').capitalize()
    
    return topics


def extract_embedding_topics(node_text, node_ids, labels, embeddings):
    """Extract topics using embedding-based centrality within each cluster."""
    print("Extracting embedding-based topics for each cluster...")
    
    # Group nodes by cluster
    cluster_nodes = defaultdict(list)
    cluster_embeddings = defaultdict(list)
    cluster_indices = defaultdict(list)
    
    for i, cluster_id in enumerate(labels):
        if i < len(node_ids):
            node_id = node_ids[i]
            cluster_nodes[cluster_id].append(node_id)
            if i < len(embeddings):
                cluster_embeddings[cluster_id].append(embeddings[i])
                cluster_indices[cluster_id].append(i)
    
    topics = {}
    for cluster_id, nodes in cluster_nodes.items():
        if not nodes or not cluster_embeddings[cluster_id]:
            topics[cluster_id] = f"Cluster {cluster_id}"
            continue
        
        # Calculate centroid of the cluster in embedding space
        centroid = np.mean(cluster_embeddings[cluster_id], axis=0)
        
        # Find node closest to centroid (most central in semantic space)
        min_distance = float('inf')
        central_idx = -1
        
        for i, emb in zip(cluster_indices[cluster_id], cluster_embeddings[cluster_id]):
            dist = np.linalg.norm(emb - centroid)
            if dist < min_distance:
                min_distance = dist
                central_idx = i
        
        if central_idx >= 0 and central_idx < len(node_ids) and node_ids[central_idx] in node_text:
            # Use the text from the most central node
            text = node_text[node_ids[central_idx]]
            
            # Extract potential topic words/phrases
            words = text.split()
            if len(words) > 5:
                # For longer text, extract important phrases
                # Remove stopwords
                filtered_words = [w for w in words if w.lower() not in ENGLISH_STOP_WORDS and len(w) > 3]
                
                # Try to extract noun phrases (simplified approach)
                # Look for consecutive capitalized words or words after "is", "are", etc.
                phrases = []
                for i in range(len(filtered_words) - 1):
                    if filtered_words[i][0].isupper() and filtered_words[i+1][0].isupper():
                        phrases.append(f"{filtered_words[i]} {filtered_words[i+1]}")
                
                if phrases:
                    topics[cluster_id] = phrases[0]
                elif filtered_words:
                    # Just use the first few important words
                    topics[cluster_id] = ' '.join(filtered_words[:3])
                else:
                    # Fall back to the first few words
                    topics[cluster_id] = ' '.join(words[:3])
            else:
                # For short text, use it directly
                topics[cluster_id] = ' '.join(words)
                
            # Clean up and capitalize
            topics[cluster_id] = re.sub(r'[^\w\s]', '', topics[cluster_id])
            topics[cluster_id] = ' '.join(word.capitalize() for word in topics[cluster_id].split())
            
            # Limit length
            if len(topics[cluster_id].split()) > 5:
                topics[cluster_id] = ' '.join(topics[cluster_id].split()[:5])
        else:
            topics[cluster_id] = f"Cluster {cluster_id}"
    
    return topics


def generate_distinct_colors(num_colors):
    """Generate distinct colors for clusters using HSV color space."""
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


def visualize_clusters(graph, node_positions, labels, cluster_topics=None, title=None, 
                      figsize=(16, 12), dpi=100, output_path=None, algorithm=None, format='png'):
    """Create and save the cluster visualization."""
    print("Creating cluster visualization...")
    
    # Convert figsize string to tuple if necessary
    if isinstance(figsize, str):
        figsize = tuple(float(x) for x in figsize.split(','))
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Generate colors for each cluster
    unique_labels = np.unique(labels)
    colors = generate_distinct_colors(len(unique_labels))
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
    
    # Create a mapping from node IDs to positions
    positions = {node_id: (node_positions[i][0], node_positions[i][1]) 
               for i, node_id in enumerate(graph.nodes()) if i < len(node_positions)}
    
    # Draw edges with light color
    edges = list(graph.edges())
    if edges:
        edge_pos = [(positions[str(u)], positions[str(v)]) for u, v in edges 
                   if str(u) in positions and str(v) in positions]
        
        if edge_pos:
            edge_collection = LineCollection(
                edge_pos, 
                colors='lightgray', 
                linewidths=0.4, 
                alpha=0.3,
                zorder=1
            )
            ax.add_collection(edge_collection)
    
    # Draw nodes colored by cluster
    for i, node in enumerate(graph.nodes()):
        if i >= len(labels):
            continue
            
        cluster_id = labels[i]
        if cluster_id == -1:  # Noise points in DBSCAN
            color = '#7f7f7f'  # Gray
        else:
            color = color_map[cluster_id]
        
        # Use world attribute count for node size if available
        node_size = 50  # Default size
        
        # Check if node has world attributes
        world_count = 0
        node_attrs = graph.nodes[node]
        for key in node_attrs:
            if key.startswith('world_') and key.split('_')[1].isdigit():
                world_count += 1
        
        if world_count > 0:
            # Scale node size based on world attribute count
            node_size = max(30, min(300, 30 + world_count * 10))
        
        x, y = positions[str(node)]
        plt.scatter(x, y, c=color, s=node_size, edgecolors='white', linewidths=0.5, alpha=0.75, zorder=2)
    
    # Add cluster labels
    if cluster_topics:
        centroids = {}
        for i, node in enumerate(graph.nodes()):
            if i >= len(labels):
                continue
                
            cluster_id = labels[i]
            if cluster_id not in centroids:
                centroids[cluster_id] = {'x': [], 'y': []}
                
            x, y = positions[str(node)]
            centroids[cluster_id]['x'].append(x)
            centroids[cluster_id]['y'].append(y)
        
        for cluster_id, coords in centroids.items():
            if not coords['x'] or not coords['y']:
                continue
                
            x = np.mean(coords['x'])
            y = np.mean(coords['y'])
            
            if cluster_id in cluster_topics:
                topic = cluster_topics[cluster_id]
                plt.annotate(
                    topic, 
                    (x, y),
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc='white',
                        ec=color_map.get(cluster_id, 'gray'),
                        alpha=0.8
                    ),
                    zorder=3
                )
    
    # # Set plot title # TODO: put title and legend visualization as an option instead
    # if title:
    #     plt.title(title, fontsize=16)
    # else:
    #     algorithm_name = algorithm.capitalize() if algorithm else "Clustering"
    #     plt.title(f"Node Clusters - {algorithm_name} Algorithm", fontsize=16)
    
    # # Add stats in bottom right
    # plt.figtext(
    #     0.95, 0.05, 
    #     f"Nodes: {graph.number_of_nodes()}\nEdges: {graph.number_of_edges()}\nClusters: {len(unique_labels)}", 
    #     ha='right', 
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    # )
    
    # Improve layout
    plt.axis('off')
    plt.tight_layout()
    
    # Save or show the figure
    if output_path:
        directory = os.path.dirname(output_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(output_path, format=format, bbox_inches='tight', dpi=dpi)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()
    
    return fig


def structural_clustering(graph, n_clusters=0, seed=42):
    """Perform clustering based on graph structure using the Louvain algorithm."""
    print("Performing structural clustering using graph topology...")
    
    # Convert to undirected for community detection
    undirected_graph = graph.to_undirected()
    
    try:
        if COMMUNITY_PACKAGE_AVAILABLE:
            # Use Louvain method
            partition = community_louvain.best_partition(undirected_graph)
            return list(graph.nodes()), [partition[node] for node in graph.nodes()]
        else:
            # Use NetworkX's greedy modularity algorithm as fallback
            communities = list(nx.algorithms.community.greedy_modularity_communities(undirected_graph))
            
            # Convert to format needed by visualization
            node_ids = list(graph.nodes())
            labels = np.zeros(len(node_ids), dtype=int)
            
            for i, community in enumerate(communities):
                for node in community:
                    idx = node_ids.index(node) if node in node_ids else -1
                    if idx >= 0:
                        labels[idx] = i
            
            return node_ids, labels
    except Exception as e:
        print(f"Error in structural clustering: {e}")
        # Fall back to spectral clustering
        adjacency_matrix = nx.to_numpy_array(graph)
        
        # Find connected components as a fallback
        if n_clusters <= 0:
            n_clusters = max(2, nx.number_connected_components(undirected_graph))
        
        clustering = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = clustering.fit_predict(adjacency_matrix)
        
        return list(graph.nodes()), labels


def main():
    """Main function."""
    args = parse_args()
    
    # Check for required packages
    if not ADVANCED_CLUSTERING_AVAILABLE and args.algorithm != "structural":
        print("Required packages not found. Please install scikit-learn and sentence-transformers.")
        print("pip install scikit-learn sentence-transformers matplotlib")
        sys.exit(1)
    
    # Load graph
    graph = load_graph(args.graph)
    
    # Extract structural features if not using text or using structural algorithm
    if args.no_text or args.algorithm == "structural":
        print("Using structural features for clustering...")
        if args.algorithm == "structural":
            # Use pure graph topology clustering
            node_ids, cluster_labels = structural_clustering(
                graph, 
                n_clusters=args.clusters,
                seed=args.seed
            )
            
            # Since we're only looking at structure, we'll need to compute T-SNE embeddings
            # from structural features for visualization
            _, structural_features = compute_structural_features(graph)
            reduced_features = reduce_dimensions(
                structural_features,
                perplexity=args.perplexity,
                learning_rate=args.learning_rate,
                iterations=args.iterations,
                seed=args.seed
            )
        else:
            # Use extracted structural features with chosen algorithm
            node_ids, features = compute_structural_features(graph)
            
            # Reduce dimensions
            reduced_features = reduce_dimensions(
                features,
                perplexity=args.perplexity,
                learning_rate=args.learning_rate,
                iterations=args.iterations,
                seed=args.seed
            )
            
            # Cluster using requested algorithm
            cluster_labels = cluster_nodes(
                features,
                algorithm=args.algorithm,
                n_clusters=args.clusters,
                seed=args.seed
            )
        
        # When using structural approach, use representative node method for topic extraction
        if args.topic_method == "representative":
            cluster_topics = extract_representative_topics(graph, node_ids, cluster_labels)
        else:
            # Default for structural clustering if not using representative method
            cluster_topics = {i: f"Cluster {i}" for i in set(cluster_labels)}
    else:
        # Extract text from node attributes
        node_text = extract_node_text(graph)
        embeddings = None
        
        if not node_text:
            print("No text attributes found in graph. Falling back to structural features.")
            # Use structural features instead
            node_ids, features = compute_structural_features(graph)
        else:
            # Generate embeddings from text
            node_ids, embeddings = generate_embeddings(node_text)
            features = embeddings
        
        # Reduce dimensions
        reduced_features = reduce_dimensions(
            features,
            perplexity=args.perplexity,
            learning_rate=args.learning_rate,
            iterations=args.iterations,
            seed=args.seed
        )
        
        # Cluster using requested algorithm
        cluster_labels = cluster_nodes(
            features if not args.cluster_on_reduced_space else reduced_features,
            algorithm=args.algorithm,
            n_clusters=args.clusters,
            seed=args.seed
        )
        
        # Extract topics for each cluster based on selected method
        if node_text:
            print(f"Using topic extraction method: {args.topic_method}")
            if args.topic_method == "frequency":
                cluster_topics = extract_cluster_topics(node_text, node_ids, cluster_labels)
            elif args.topic_method == "ngram":
                cluster_topics = extract_ngram_topics(node_text, node_ids, cluster_labels)
            elif args.topic_method == "representative":
                cluster_topics = extract_representative_topics(graph, node_ids, cluster_labels)
            elif args.topic_method == "embedding":
                if embeddings is not None:
                    cluster_topics = extract_embedding_topics(node_text, node_ids, cluster_labels, embeddings)
                else:
                    print("Warning: Embeddings not available, falling back to frequency method")
                    cluster_topics = extract_cluster_topics(node_text, node_ids, cluster_labels)
            else:
                # Default fallback
                cluster_topics = extract_cluster_topics(node_text, node_ids, cluster_labels)
        else:
            cluster_topics = {i: f"Cluster {i}" for i in set(cluster_labels)}
    
    # Create output path with timestamp if not specified
    if args.output:
        output_path = args.output
        if not output_path.endswith(f".{args.format}"):
            output_path = f"{output_path}.{args.format}"
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = f"../data/figures/clusters_{timestamp}.{args.format}"
    
    # Create visualization
    fig = visualize_clusters(
        graph,
        reduced_features,
        cluster_labels,
        cluster_topics=cluster_topics,
        title=args.title,
        figsize=args.figsize,
        dpi=args.dpi,
        output_path=output_path,
        algorithm=args.algorithm, 
        format=args.format
    )
    
    print("Done!")


if __name__ == "__main__":
    main()