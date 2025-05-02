// Global variables
let network = null;
let graphData = null;
let nodesDataset = null;
let edgesDataset = null;
let selectedElement = null;
let physicsEnabled = true;
let highlightedNodes = new Set();
let highlightedEdges = new Set(); // Track highlighted edges
let worldKeys = [];
let originalNodeColors = new Map(); // Store original node colors

// DOM elements
const fileInput = document.getElementById('graph-file');
const fileName = document.getElementById('file-name');
const graphContainer = document.getElementById('graph-container');
const loadingOverlay = document.getElementById('loading');
const infoPanel = document.getElementById('info-panel');
const closeInfoBtn = document.getElementById('close-info');
const infoTitle = document.getElementById('info-title');
const attributesContainer = document.getElementById('attributes-container');
const graphStats = document.getElementById('graph-stats');
const zoomInBtn = document.getElementById('zoom-in');
const zoomOutBtn = document.getElementById('zoom-out');
const zoomFitBtn = document.getElementById('zoom-fit');
const togglePhysicsBtn = document.getElementById('toggle-physics');
const clusterLegend = document.getElementById('cluster-legend');
const clusterLegendItems = document.getElementById('cluster-legend-items');
const toggleLegendBtn = document.getElementById('toggle-legend');

// Search related elements
const worldKeySelect = document.getElementById('world-key-select');
const searchAttributeBtn = document.getElementById('search-attribute-btn');
const clearSearchBtn = document.getElementById('clear-search-btn');
const searchResults = document.getElementById('search-results');
const resultsCount = document.getElementById('results-count');

// Add event listeners for analysis buttons
const analysisPanel = document.getElementById('analysis-panel');
const toggleAnalysisBtn = document.getElementById('toggle-analysis');
const closeAnalysisBtn = document.getElementById('close-analysis');
const analysisTabs = document.querySelectorAll('.tab-button');
const analysisContents = document.querySelectorAll('.tab-content');

// Analysis buttons
const analyzeStructuralBtn = document.getElementById('analyze-structural');
const analyzeCausalBtn = document.getElementById('analyze-causal');
const analyzeCommunityBtn = document.getElementById('analyze-community');
const analyzeCentralityBtn = document.getElementById('analyze-centrality');
const analyzeDomainBtn = document.getElementById('analyze-domain');

// Add event listener for the topic analysis button
const analyzeTopicBtn = document.getElementById('analyze-topic');
analyzeTopicBtn.addEventListener('click', () => fetchAnalysis('topic', 'topic-results'));

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    fileInput.addEventListener('change', handleFileSelect);
    closeInfoBtn.addEventListener('click', closeInfoPanel);
    zoomInBtn.addEventListener('click', () => zoomNetwork(0.2));
    zoomOutBtn.addEventListener('click', () => zoomNetwork(-0.2));
    zoomFitBtn.addEventListener('click', fitNetwork);
    togglePhysicsBtn.addEventListener('click', togglePhysics);
    toggleLegendBtn.addEventListener('click', toggleLegend);
    
    // World search is now directly triggered on select change
    clearSearchBtn.addEventListener('click', clearWorldHighlight);
    worldKeySelect.addEventListener('change', worldKeySelectChanged);
    
    // Check if graph path was provided as a URL parameter
    const urlParams = new URLSearchParams(window.location.search);
    
    // Check if we should load a startup graph (provided via command line)
    if (urlParams.has('loadStartupGraph')) {
        loadStartupGraph();
    }
    // Otherwise, check for direct graph parameter
    else if (urlParams.has('graph')) {
        loadGraphFromPath(urlParams.get('graph'));
    }

    // Event listener to toggle the analysis panel
    toggleAnalysisBtn.addEventListener('click', () => {
        analysisPanel.classList.toggle('visible');
    });

    // Event listener to close the analysis panel
    closeAnalysisBtn.addEventListener('click', () => {
        analysisPanel.classList.remove('visible');
    });

    // Event listeners for tab switching
    analysisTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            analysisTabs.forEach(t => t.classList.remove('active'));
            analysisContents.forEach(content => content.classList.remove('active'));

            // Add active class to the clicked tab and corresponding content
            tab.classList.add('active');
            const target = document.getElementById(`${tab.dataset.tab}-tab`);
            if (target) target.classList.add('active');
        });
    });

    // Event listeners for analysis buttons
    analyzeStructuralBtn.addEventListener('click', () => fetchAnalysis('structural', 'structural-results'));
    analyzeCausalBtn.addEventListener('click', () => fetchAnalysis('causal', 'causal-results'));
    analyzeCommunityBtn.addEventListener('click', () => fetchAnalysis('community', 'community-results'));
    analyzeCentralityBtn.addEventListener('click', () => fetchAnalysis('centrality', 'centrality-results'));
    analyzeDomainBtn.addEventListener('click', () => fetchAnalysis('domain', 'domain-results'));
});

/**
 * Handle file selection from the file input
 */
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    fileName.textContent = file.name;
    loadGraphFromFile(file);
}

/**
 * Loads a graph from a file
 */
function loadGraphFromFile(file) {
    showLoading();
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/api/load-graph', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        graphData = data;
        visualizeGraph(data);
        populateWorldKeySelect(data.worldKeys || []);
        hideLoading();
    })
    .catch(error => {
        console.error('Error loading graph:', error);
        alert('Error loading graph: ' + error.message);
        hideLoading();
    });
}

/**
 * Loads a graph from a path (used for command-line arguments)
 */
function loadGraphFromPath(path) {
    showLoading();
    
    fetch('/api/load-graph-path', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ path })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        fileName.textContent = path.split(/[\\/]/).pop();
        graphData = data;
        visualizeGraph(data);
        populateWorldKeySelect(data.worldKeys || []);
        hideLoading();
    })
    .catch(error => {
        console.error('Error loading graph:', error);
        alert('Error loading graph: ' + error.message);
        hideLoading();
    });
}

/**
 * Load a graph that was specified at startup via command line
 */
function loadStartupGraph() {
    showLoading();
    
    fetch('/api/startup-graph')
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            graphData = data;
            visualizeGraph(data);
            populateWorldKeySelect(data.worldKeys || []);
            hideLoading();
        })
        .catch(error => {
            console.error('Error loading startup graph:', error);
            alert('Error loading startup graph: ' + error.message);
            hideLoading();
        });
}

/**
 * Populate the world key select dropdown
 */
function populateWorldKeySelect(keys) {
    // Store world keys for later use
    worldKeys = keys;
    
    // Clear the select
    worldKeySelect.innerHTML = '<option value="">Select a world_i...</option>';
    
    // Add options for each world key
    keys.forEach(key => {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = key;
        worldKeySelect.appendChild(option);
    });
    
    // Enable/disable elements based on whether we have world keys
    const hasWorldKeys = keys.length > 0;
    worldKeySelect.disabled = !hasWorldKeys;
    clearSearchBtn.disabled = !hasWorldKeys;
}

/**
 * Enable search button when a world key is selected
 */
function enableSearchButton() {
    const isWorldKeySelected = worldKeySelect.value !== '';
    searchAttributeBtn.disabled = !isWorldKeySelected;
}

/**
 * Handle world key selection change
 */
function worldKeySelectChanged() {
    const worldKey = worldKeySelect.value;
    if (worldKey) {
        highlightNodesWithWorld(worldKey);
    } else {
        clearWorldHighlight();
    }
}

/**
 * Highlight nodes that have the selected world_i attribute
 */
function highlightNodesWithWorld(worldKey) {
    if (!nodesDataset) return;
    
    fetch('/api/search-world-attribute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            worldKey
        })
    })
    .then(response => response.json())
    .then(data => {
        const matchingNodes = new Set(data.matchingNodes || []);
        
        // Store original colors first time
        if (originalNodeColors.size === 0) {
            storeOriginalNodeColors();
        }

        // Update all node colors
        const allNodes = nodesDataset.get();
        const updates = [];
        
        allNodes.forEach(node => {
            if (matchingNodes.has(node.id)) {
                // Bright color for matching nodes
                updates.push({
                    id: node.id,
                    color: {
                        background: brightenColor(originalNodeColors.get(node.id).background, 0.7),
                        border: '#34495e',
                        highlight: {
                            background: brightenColor(originalNodeColors.get(node.id).background, 0.9),
                            border: '#2c3e50'
                        }
                    },
                    borderWidth: 3
                });
            } else {
                // Dull color for non-matching nodes
                updates.push({
                    id: node.id,
                    color: {
                        background: dullColor(originalNodeColors.get(node.id).background, 0.5),
                        border: '#bdc3c7',
                        highlight: {
                            background: dullColor(originalNodeColors.get(node.id).background, 0.3),
                            border: '#95a5a6'
                        }
                    },
                    borderWidth: 1
                });
            }
        });
        
        nodesDataset.update(updates);
        
        // Update results display
        resultsCount.textContent = matchingNodes.size;
        searchResults.classList.remove('hidden');
        
        if (matchingNodes.size > 0) {
            // Focus the view on the matching nodes
            network.fit({
                nodes: Array.from(matchingNodes),
                animation: {
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }
    })
    .catch(error => {
        console.error('Error searching for world attributes:', error);
    });
}

/**
 * Store the original colors of all nodes for later restoration
 */
function storeOriginalNodeColors() {
    if (!nodesDataset) return;
    
    const allNodes = nodesDataset.get();
    allNodes.forEach(node => {
        originalNodeColors.set(node.id, {
            background: node.color.background,
            border: node.color.border,
            highlight: node.color.highlight
        });
    });
}

/**
 * Clear world highlighting and restore original colors
 */
function clearWorldHighlight() {
    if (!nodesDataset || originalNodeColors.size === 0) return;
    
    // Reset world selection
    worldKeySelect.selectedIndex = 0;
    
    // Reset node colors to original
    const updates = [];
    originalNodeColors.forEach((color, nodeId) => {
        updates.push({
            id: nodeId,
            color: {
                background: color.background,
                border: color.border,
                highlight: color.highlight
            },
            borderWidth: 1
        });
    });
    
    nodesDataset.update(updates);
    searchResults.classList.add('hidden');
}

/**
 * Brighten a color by a factor
 */
function brightenColor(hexColor, factor) {
    // Convert hex to RGB
    let r = parseInt(hexColor.slice(1, 3), 16);
    let g = parseInt(hexColor.slice(3, 5), 16);
    let b = parseInt(hexColor.slice(5, 7), 16);
    
    // Brighten
    r = Math.min(255, Math.floor(r + (255 - r) * factor));
    g = Math.min(255, Math.floor(g + (255 - g) * factor));
    b = Math.min(255, Math.floor(b + (255 - b) * factor));
    
    // Convert back to hex
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/**
 * Dull a color by reducing saturation and brightness
 */
function dullColor(hexColor, factor) {
    // Convert hex to RGB
    let r = parseInt(hexColor.slice(1, 3), 16);
    let g = parseInt(hexColor.slice(3, 5), 16);
    let b = parseInt(hexColor.slice(5, 7), 16);
    
    // Calculate luminance (brightness)
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    
    // Mix with gray to dull
    r = Math.floor(r * factor + luminance * (1 - factor));
    g = Math.floor(g * factor + luminance * (1 - factor));
    b = Math.floor(b * factor + luminance * (1 - factor));
    
    // Convert back to hex
    return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
}

/**
 * Search for nodes with specific world attributes
 */
function searchWorldAttribute() {
    const worldKey = worldKeySelect.value;
    
    if (!worldKey) {
        alert('Please select a world to search for');
        return;
    }
    
    fetch('/api/search-world-attribute', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            worldKey,
            attributeKey: '*',  // Wildcard to find any nodes with this world_i
            attributeValue: null
        })
    })
    .then(response => response.json())
    .then(data => {
        // Clear previous highlights
        clearNodeHighlights();
        
        const matchingNodes = data.matchingNodes || [];
        
        // Highlight matching nodes
        highlightNodes(matchingNodes);
        
        // Update results display
        resultsCount.textContent = matchingNodes.length;
        searchResults.classList.remove('hidden');
        
        if (matchingNodes.length > 0) {
            // Focus the view on the matching nodes
            const nodeIds = matchingNodes;
            network.fit({
                nodes: nodeIds,
                animation: {
                    duration: 1000,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }
    })
    .catch(error => {
        console.error('Error searching for attributes:', error);
        alert('Error searching for attributes: ' + error.message);
    });
}

/**
 * Clear search results and node highlights
 */
function clearSearch() {
    clearNodeHighlights();
    searchResults.classList.add('hidden');
    worldKeySelect.selectedIndex = 0;
    searchAttributeBtn.disabled = true;
}

/**
 * Clear all node and edge highlights
 */
function clearNodeHighlights() {
    // Clear highlighted nodes
    if (nodesDataset) {
        highlightedNodes.forEach(nodeId => {
            const node = nodesDataset.get(nodeId);
            if (node) {
                nodesDataset.update({
                    id: nodeId,
                    borderWidth: 1,
                    shadow: {
                        enabled: false
                    }
                });
            }
        });
        highlightedNodes.clear();
    }
    
    // Clear highlighted edges
    if (edgesDataset && highlightedEdges.size > 0) {
        highlightedEdges.forEach(edgeId => {
            const edge = edgesDataset.get(edgeId);
            if (edge) {
                edgesDataset.update({
                    id: edgeId,
                    width: 2,
                    color: undefined // Reset to default color
                });
            }
        });
        highlightedEdges.clear();
    }
}

/**
 * Highlight specific nodes and edges
 * @param {Array} nodeIds - Array of node IDs to highlight
 * @param {Array} edgeIds - Array of edge IDs to highlight (optional)
 */
function highlightNodes(nodeIds, edgeIds = []) {
    if (nodesDataset) {
        // Highlight nodes
        nodeIds.forEach(nodeId => {
            highlightedNodes.add(nodeId);
            const node = nodesDataset.get(nodeId);
            if (node) {
                nodesDataset.update({
                    id: nodeId,
                    borderWidth: 3,
                    shadow: {
                        enabled: true,
                        color: 'rgba(231, 76, 60, 0.5)',
                        size: 10
                    }
                });
            }
        });
        
        // Highlight specific edges if provided
        if (edgeIds && edgeIds.length > 0 && edgesDataset) {
            // Store which edges are highlighted
            highlightedEdges = new Set(edgeIds);
            
            // Update the edges to highlight them
            edgeIds.forEach(edgeId => {
                const edge = edgesDataset.get(edgeId);
                if (edge) {
                    edgesDataset.update({
                        id: edgeId,
                        width: 4,
                        color: {
                            color: '#e74c3c',
                            highlight: '#e74c3c',
                            hover: '#c0392b'
                        }
                    });
                }
            });
        }
    }
}

/**
 * Find edges that form a path between sequential nodes
 * @param {Array} nodePath - Ordered array of node IDs forming a path
 * @returns {Array} - Array of edge IDs forming the path
 */
function findPathEdges(nodePath) {
    if (!edgesDataset || !nodePath || nodePath.length < 2) {
        return [];
    }
    
    const pathEdges = [];
    const allEdges = edgesDataset.get();
    
    // Create a map for faster lookup of edges between nodes
    const edgeMap = new Map();
    allEdges.forEach(edge => {
        const key = `${edge.from}->${edge.to}`;
        edgeMap.set(key, edge.id);
    });
    
    // For each consecutive pair of nodes in the path, find the connecting edge
    for (let i = 0; i < nodePath.length - 1; i++) {
        const fromNode = nodePath[i];
        const toNode = nodePath[i + 1];
        const edgeKey = `${fromNode}->${toNode}`;
        
        const edgeId = edgeMap.get(edgeKey);
        if (edgeId) {
            pathEdges.push(edgeId);
        }
    }
    
    return pathEdges;
}

/**
 * Creates and displays the graph visualization
 */
function visualizeGraph(data) {
    if (network !== null) {
        network.destroy();
        network = null;
    }
    
    // Create the legend for clusters
    createClusterLegend(data.clusters || []);
    
    // Prepare nodes with enhanced features
    const nodes = data.nodes.map(node => {
        const nodeData = {
            id: node.id,
            label: node.label,
            title: node.title || node.label,
            group: node.group || 0, // For cluster-based coloring
            attributes: node.attributes || {},
            shape: 'ellipse'
        };
        
        // If node has custom position, use it
        if (node.x !== undefined && node.y !== undefined) {
            nodeData.x = node.x;
            nodeData.y = node.y;
        }
        
        // If node has custom color, use it
        if (node.color) {
            nodeData.color = node.color;
        }
        
        // Special border for nodes with world attributes
        if (node.hasWorldAttrs) {
            if (!nodeData.borderWidth) {
                nodeData.borderWidth = 2;
            }
            nodeData.shapeProperties = {
                borderDashes: [5, 5] // Dashed border
            };
        }
        
        return nodeData;
    });
    
    // Prepare edges with attributes
    const edges = data.edges.map(edge => {
        const edgeData = {
            id: edge.id || `${edge.from}-${edge.to}`,
            from: edge.from,
            to: edge.to,
            attributes: edge.attributes || {},
            arrows: edge.arrows || (data.directed ? 'to' : ''),
            title: edge.title || `Edge: ${edge.from} → ${edge.to}`
        };
        
        // If edge has custom color, use it
        if (edge.color) {
            edgeData.color = edge.color;
        }
        
        return edgeData;
    });
    
    // Create vis.js datasets
    nodesDataset = new vis.DataSet(nodes);
    edgesDataset = new vis.DataSet(edges);
    
    const visData = {
        nodes: nodesDataset,
        edges: edgesDataset
    };
    
    // Configuration for the network with better stability and less stacking
    const options = {
        nodes: {
            shape: 'ellipse',
            borderWidth: 1,
            size: 25,
            font: {
                size: 14,
                face: 'Segoe UI'
            }
        },
        edges: {
            width: 2,
            smooth: {
                enabled: true,
                type: 'continuous', // More stable than dynamic
                roundness: 0.5
            },
            arrows: {
                to: {
                    enabled: data.directed,
                    scaleFactor: 0.5
                }
            }
        },
        physics: {
            enabled: true,
            stabilization: {
                iterations: 1000,
                updateInterval: 50,
                fit: true
            },
            barnesHut: {
                gravitationalConstant: -5000, // More repulsion (was -3000)
                centralGravity: 0.3,         // Lower to allow more spreading (was 0.5)
                springLength: 200,           // Longer springs to spread nodes (was 150)
                springConstant: 0.01,        // Less stiff edges for wider spacing (was 0.02)
                damping: 0.4,                // Same damping
                avoidOverlap: 0.8            // Higher value to prevent stacking
            },
            timestep: 0.5,                   // Same timestep
            minVelocity: 0.75                // Same velocity threshold
        },
        interaction: {
            hover: true,
            tooltipDelay: 300,
            zoomView: true,
            dragView: true,
            multiselect: true,
            navigationButtons: true,
            keyboard: true
        },
        groups: {} // Will be populated from cluster colors
    };
    
    // Set up color groups for clusters
    if (data.clusters) {
        data.clusters.forEach(cluster => {
            options.groups[cluster.id] = {
                color: {
                    background: cluster.color,
                    border: '#2c3e50',
                    highlight: {
                        background: cluster.color,
                        border: '#34495e'
                    }
                }
            };
        });
    }
    
    // Create the network
    network = new vis.Network(graphContainer, visData, options);
    
    // Update statistics
    updateGraphStats(data.nodes.length, data.edges.length);
    
    // Event listeners for the network
    setUpNetworkEvents();
}

/**
 * Set up event listeners for the network visualization
 */
function setUpNetworkEvents() {
    if (!network) return;
    
    // Click event to show node/edge info
    network.on('click', handleNetworkClick);
    
    // Stabilization progress
    network.on('stabilizationProgress', function(params) {
        const progress = Math.round(params.iterations / params.total * 100);
        loadingOverlay.innerHTML = `<div class="loading-spinner"></div>
                                    <p>Stabilizing network... ${progress}%</p>
                                    <p>This ensures graph stability for large networks</p>`;
        loadingOverlay.classList.remove('hidden');
    });
    
    // Stabilization complete
    network.on('stabilizationIterationsDone', function() {
        loadingOverlay.classList.add('hidden');
        
        // Physics can be a bit unstable at first, run a final fit after stabilization
        setTimeout(() => {
            if (network) network.fit();
        }, 500);
    });
    
    // Double click to fit the view to the clicked node's neighborhood
    network.on('doubleClick', function(params) {
        if (params.nodes.length > 0) {
            const nodeId = params.nodes[0];
            const connectedNodes = network.getConnectedNodes(nodeId);
            network.fit({
                nodes: [...connectedNodes, nodeId],
                animation: {
                    duration: 800,
                    easingFunction: 'easeInOutQuad'
                }
            });
        }
    });
}

/**
 * Create the cluster legend
 */
function createClusterLegend(clusters) {
    // Clear any existing legend items
    clusterLegendItems.innerHTML = '';
    
    clusters.forEach(cluster => {
        const item = document.createElement('div');
        item.className = 'cluster-legend-item';
        
        const colorSwatch = document.createElement('div');
        colorSwatch.className = 'color-swatch';
        colorSwatch.style.backgroundColor = cluster.color;
        
        const label = document.createElement('span');
        label.textContent = `Cluster ${cluster.id}`;
        
        item.appendChild(colorSwatch);
        item.appendChild(label);
        clusterLegendItems.appendChild(item);
    });
}

/**
 * Toggle the visibility of the cluster legend
 */
function toggleLegend() {
    clusterLegend.classList.toggle('hidden');
    toggleLegendBtn.textContent = clusterLegend.classList.contains('hidden') 
                              ? 'Show Cluster Legend' 
                              : 'Hide Cluster Legend';
}

/**
 * Toggle physics simulation on/off
 */
function togglePhysics() {
    physicsEnabled = !physicsEnabled;
    
    if (network) {
        network.setOptions({ physics: { enabled: physicsEnabled } });
    }
    
    togglePhysicsBtn.classList.toggle('active');
    togglePhysicsBtn.title = physicsEnabled ? 'Disable Physics' : 'Enable Physics';
}

/**
 * Handle network element click
 */
function handleNetworkClick(params) {
    if (params.nodes.length > 0) {
        // Node clicked
        const nodeId = params.nodes[0];
        const node = nodesDataset.get(nodeId);
        
        if (node) {
            // Extract nested world attributes for better display
            const displayAttributes = processNodeAttributes(node.attributes);
            showInfoPanel('Node: ' + node.label, displayAttributes);
            selectedElement = { type: 'node', data: node };
        }
    } else if (params.edges.length > 0) {
        // Edge clicked
        const edgeId = params.edges[0];
        const edge = edgesDataset.get(edgeId);
        
        if (edge) {
            // Get the source and target nodes for better labeling
            const sourceNode = nodesDataset.get(edge.from);
            const targetNode = nodesDataset.get(edge.to);
            
            const sourceLabel = sourceNode ? sourceNode.label : edge.from;
            const targetLabel = targetNode ? targetNode.label : edge.to;
            
            showInfoPanel(
                `Edge: ${sourceLabel} → ${targetLabel}`,
                edge.attributes
            );
            selectedElement = { type: 'edge', data: edge };
        }
    } else {
        // Background clicked
        closeInfoPanel();
        selectedElement = null;
    }
}

/**
 * Process node attributes to better display nested objects like world_i
 */
function processNodeAttributes(attributes) {
    const processed = {};
    
    // Process each attribute
    for (const key in attributes) {
        const value = attributes[key];
        
        // Check if this is a world_i attribute
        if (key.startsWith('world_') && key.split('_')[1].isdigit) {
            try {
                // Try to parse if it's a string representation of an object
                if (typeof value === 'string' && value.startsWith('{') && value.endsWith('}')) {
                    processed[key] = JSON.parse(value.replace(/'/g, '"'));
                } else {
                    processed[key] = value;
                }
            } catch (e) {
                processed[key] = value;
            }
        } else {
            processed[key] = value;
        }
    }
    
    return processed;
}

/**
 * Display the info panel with attributes
 */
function showInfoPanel(title, attributes) {
    infoTitle.textContent = title;
    attributesContainer.innerHTML = '';
    
    // Sort attributes by key
    const sortedKeys = Object.keys(attributes || {}).sort();
    
    if (sortedKeys.length === 0) {
        attributesContainer.innerHTML = '<p>No attributes available</p>';
    } else {
        for (const key of sortedKeys) {
            const value = attributes[key];
            
            const attributeItem = document.createElement('div');
            attributeItem.className = 'attribute-item';
            
            const attributeName = document.createElement('div');
            attributeName.className = 'attribute-name';
            attributeName.textContent = key;
            
            const attributeValue = document.createElement('div');
            attributeValue.className = 'attribute-value';
            
            // Format value based on type
            if (typeof value === 'object' && value !== null) {
                attributeValue.textContent = JSON.stringify(value, null, 2);
            } else {
                attributeValue.textContent = value;
            }
            
            attributeItem.appendChild(attributeName);
            attributeItem.appendChild(attributeValue);
            attributesContainer.appendChild(attributeItem);
        }
    }
    
    infoPanel.classList.remove('hidden');
    infoPanel.classList.add('visible');
}

/**
 * Close the info panel
 */
function closeInfoPanel() {
    infoPanel.classList.remove('visible');
    setTimeout(() => {
        infoPanel.classList.add('hidden');
    }, 300);
}

/**
 * Update graph statistics display
 */
function updateGraphStats(nodeCount, edgeCount) {
    graphStats.textContent = `${nodeCount} nodes, ${edgeCount} edges`;
}

/**
 * Show loading overlay
 */
function showLoading() {
    loadingOverlay.innerHTML = `<div class="loading-spinner"></div><p>Loading graph...</p>`;
    loadingOverlay.classList.remove('hidden');
}

/**
 * Hide loading overlay
 */
function hideLoading() {
    loadingOverlay.classList.add('hidden');
}

/**
 * Zoom the network view
 */
function zoomNetwork(scale) {
    if (network) {
        const currentScale = network.getScale();
        network.moveTo({
            scale: currentScale * (1 + scale)
        });
    }
}

/**
 * Fit all nodes in the view
 */
function fitNetwork() {
    if (network) {
        network.fit({
            animation: {
                duration: 1000,
                easingFunction: 'easeInOutQuad'
            }
        });
    }
}

/**
 * Function to fetch and display analysis results
 */
function fetchAnalysis(type, resultContainerId) {
    if (!graphData) {
        alert('No graph loaded. Please load a graph first.');
        return;
    }

    const resultContainer = document.getElementById(resultContainerId);
    resultContainer.innerHTML = '<p>Loading...</p>';

    fetch('/api/analyze-graph', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ type })
    })
    .then(response => {
        // Always attempt to parse the JSON, even for error responses
        return response.json().then(data => {
            // If response is not ok, but we have data, use the error details from data
            if (!response.ok) {
                throw { 
                    message: data.error || `HTTP error ${response.status}`,
                    details: data,
                    status: response.status
                };
            }
            return data;
        });
    })
    .then(data => {
        renderAnalysisResults(data, resultContainer);
    })
    .catch(error => {
        console.error('Error fetching analysis results:', error);
        
        // Create a more detailed error display
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        
        const errorTitle = document.createElement('h3');
        errorTitle.textContent = 'Analysis Error';
        errorDiv.appendChild(errorTitle);
        
        // Display the main error message
        const errorText = document.createElement('p');
        errorText.textContent = error.message || 'Failed to fetch analysis results';
        errorDiv.appendChild(errorText);
        
        // Display error details if available
        if (error.details) {
            const detailsContainer = document.createElement('div');
            detailsContainer.className = 'error-details-container';
            
            const detailsToggle = document.createElement('button');
            detailsToggle.textContent = 'Show Error Details';
            detailsToggle.className = 'error-details-toggle';
            
            const detailsContent = document.createElement('pre');
            detailsContent.className = 'error-details-content';
            detailsContent.style.display = 'none';
            
            // Format error details nicely
            if (error.details.error_type) {
                detailsContent.textContent = `Error Type: ${error.details.error_type}\n\n`;
                
                if (error.details.traceback) {
                    detailsContent.textContent += `Traceback:\n${error.details.traceback}`;
                }
                
                // Add debug info if available
                if (error.details.debug_info) {
                    detailsContent.textContent += '\nDebug Info:\n';
                    detailsContent.textContent += JSON.stringify(error.details.debug_info, null, 2);
                }
            } else {
                detailsContent.textContent = JSON.stringify(error.details, null, 2);
            }
            
            detailsToggle.addEventListener('click', () => {
                if (detailsContent.style.display === 'none') {
                    detailsContent.style.display = 'block';
                    detailsToggle.textContent = 'Hide Error Details';
                } else {
                    detailsContent.style.display = 'none';
                    detailsToggle.textContent = 'Show Error Details';
                }
            });
            
            detailsContainer.appendChild(detailsToggle);
            detailsContainer.appendChild(detailsContent);
            errorDiv.appendChild(detailsContainer);
        }
        
        // Add troubleshooting suggestions
        const errorHelp = document.createElement('div');
        errorHelp.className = 'error-help';
        errorHelp.innerHTML = `<p>Troubleshooting suggestions:</p>
            <ul>
                <li>Try a different analysis type</li>
                <li>Verify the graph structure is valid</li>
                <li>Check if the graph has the required attributes for this analysis type</li>
                <li>Restart the server with the --debug flag for more detailed error information</li>
                <li>Reload the graph and try again</li>
            </ul>`;
        errorDiv.appendChild(errorHelp);
        
        resultContainer.innerHTML = '';
        resultContainer.appendChild(errorDiv);
    });
}

/**
 * Function to render analysis results with visualizations
 */
function renderAnalysisResults(data, container) {
    container.innerHTML = ''; // Clear previous results

    if (typeof data !== 'object' || data === null) {
        container.innerHTML = '<p>No data available</p>';
        return;
    }

    // Different rendering logic based on the type of analysis
    if (data.density !== undefined) {
        // Structural properties
        renderStructuralAnalysis(data, container);
    } else if (data.longest_chain !== undefined) {
        // Causal properties
        renderCausalAnalysis(data, container);
    } else if (data.bridging_nodes !== undefined) {
        // Community structure
        renderCommunityAnalysis(data, container);
    } else if (data.topic_clusters !== undefined) {
        // Topic clusters
        renderTopicAnalysis(data, container);
    } else {
        // Generic fallback rendering for any other data
        for (const [key, value] of Object.entries(data)) {
            const card = document.createElement('div');
            card.className = 'metric-card';

            const header = document.createElement('div');
            header.className = 'metric-header';
            header.textContent = formatKey(key);

            const content = document.createElement('div');
            content.className = 'metric-value';

            if (typeof value === 'object' && value !== null) {
                content.textContent = JSON.stringify(value, null, 2);
            } else {
                content.textContent = value;
            }

            card.appendChild(header);
            card.appendChild(content);
            container.appendChild(card);
        }
    }
}

/**
 * Render structural analysis results with visualizations
 */
function renderStructuralAnalysis(data, container) {
    // Density
    const densityCard = createMetricCard('Graph Density', data.density.toFixed(6));
    container.appendChild(densityCard);
    
    // Components
    const componentsCard = createMetricCard('Components', '');
    const componentsContent = componentsCard.querySelector('.metric-value');
    
    for (const [key, value] of Object.entries(data.components)) {
        const p = document.createElement('p');
        p.textContent = `${formatKey(key)}: ${value}`;
        componentsContent.appendChild(p);
    }
    container.appendChild(componentsCard);
    
    // Path metrics
    const pathCard = createMetricCard('Path Metrics', '');
    const pathContent = pathCard.querySelector('.metric-value');
    
    for (const [key, value] of Object.entries(data.path_metrics)) {
        const p = document.createElement('p');
        p.textContent = `${formatKey(key)}: ${value}`;
        pathContent.appendChild(p);
    }
    container.appendChild(pathCard);
    
    // In-Degree Distribution Chart
    if (data.in_degree_distribution && data.in_degree_distribution.labels && data.in_degree_distribution.labels.length > 0) {
        const inDegreeCard = document.createElement('div');
        inDegreeCard.className = 'metric-card';
        
        const inDegreeHeader = document.createElement('div');
        inDegreeHeader.className = 'metric-header';
        inDegreeHeader.textContent = 'In-Degree Distribution';
        
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.style.height = '250px';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'in-degree-chart';
        chartContainer.appendChild(canvas);
        
        inDegreeCard.appendChild(inDegreeHeader);
        inDegreeCard.appendChild(chartContainer);
        container.appendChild(inDegreeCard);
        
        // Create chart
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.in_degree_distribution.labels,
                datasets: [{
                    label: 'Number of nodes',
                    data: data.in_degree_distribution.values,
                    backgroundColor: '#3498db',
                    borderColor: '#2980b9',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Node Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'In-Degree Value'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Distribution of In-Degree Values'
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return `In-Degree: ${tooltipItems[0].label}`;
                            },
                            label: function(context) {
                                return `${context.raw} nodes`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Out-Degree Distribution Chart
    if (data.out_degree_distribution && data.out_degree_distribution.labels && data.out_degree_distribution.labels.length > 0) {
        const outDegreeCard = document.createElement('div');
        outDegreeCard.className = 'metric-card';
        
        const outDegreeHeader = document.createElement('div');
        outDegreeHeader.className = 'metric-header';
        outDegreeHeader.textContent = 'Out-Degree Distribution';
        
        const chartContainer = document.createElement('div');
        chartContainer.className = 'chart-container';
        chartContainer.style.height = '250px';
        
        const canvas = document.createElement('canvas');
        canvas.id = 'out-degree-chart';
        chartContainer.appendChild(canvas);
        
        outDegreeCard.appendChild(outDegreeHeader);
        outDegreeCard.appendChild(chartContainer);
        container.appendChild(outDegreeCard);
        
        // Create chart
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.out_degree_distribution.labels,
                datasets: [{
                    label: 'Number of nodes',
                    data: data.out_degree_distribution.values,
                    backgroundColor: '#e74c3c',
                    borderColor: '#c0392b',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Node Count'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Out-Degree Value'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Distribution of Out-Degree Values'
                    },
                    tooltip: {
                        callbacks: {
                            title: function(tooltipItems) {
                                return `Out-Degree: ${tooltipItems[0].label}`;
                            },
                            label: function(context) {
                                return `${context.raw} nodes`;
                            }
                        }
                    }
                }
            }
        });
    }
}

/**
 * Render causal analysis results
 */
function renderCausalAnalysis(data, container) {
    // Root causes and terminal effects counts
    const causesCard = createMetricCard('Causal Sources and Targets', '');
    const causesContent = causesCard.querySelector('.metric-value');
    
    const rootCausesP = document.createElement('p');
    rootCausesP.textContent = `Root Causes: ${data.root_causes_count}`;
    causesContent.appendChild(rootCausesP);
    
    const terminalEffectsP = document.createElement('p');
    terminalEffectsP.textContent = `Terminal Effects: ${data.terminal_effects_count}`;
    causesContent.appendChild(terminalEffectsP);
    
    container.appendChild(causesCard);
    
    // Longest chain
    if (data.longest_chain && data.longest_chain.length > 0) {
        const longestChainCard = createMetricCard('Longest Causal Chain', '');
        const longestChainContent = longestChainCard.querySelector('.metric-value');
        
        const chainLength = document.createElement('p');
        chainLength.textContent = `Length: ${data.longest_chain_length}`;
        longestChainContent.appendChild(chainLength);
        
        const chainPath = document.createElement('div');
        chainPath.className = 'path-display';
        chainPath.textContent = data.longest_chain.join(' → ');
        longestChainContent.appendChild(chainPath);
        
        // Add button to highlight this path in the graph
        const highlightBtn = document.createElement('button');
        highlightBtn.className = 'highlight-button';
        highlightBtn.textContent = 'Highlight Chain';
        
        // Track if this chain is currently highlighted
        let isHighlighted = false;
        
        highlightBtn.addEventListener('click', () => {
            if (network) {
                if (isHighlighted) {
                    // If already highlighted, clear highlights
                    clearNodeHighlights();
                    highlightBtn.textContent = 'Highlight Chain';
                    isHighlighted = false;
                } else {
                    // Clear any existing highlights
                    clearNodeHighlights();
                    
                    // Find the exact edges that form the path
                    const pathEdges = findPathEdges(data.longest_chain);
                    
                    // Highlight both nodes and edges in the chain
                    highlightNodes(data.longest_chain, pathEdges);
                    
                    // Update button text
                    highlightBtn.textContent = 'Remove Highlight';
                    isHighlighted = true;
                    
                    // Focus on these nodes
                    network.fit({
                        nodes: data.longest_chain,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            }
        });
        
        longestChainContent.appendChild(highlightBtn);
        
        container.appendChild(longestChainCard);
    }
    
    // Feedback loops count
    const feedbackCard = createMetricCard('Feedback Loops', data.feedback_loops_count);
    container.appendChild(feedbackCard);
    
    // Causal bottlenecks
    if (data.bottlenecks && data.bottlenecks.length > 0) {
        const bottlenecksCard = createMetricCard('Causal Bottlenecks', '');
        const bottlenecksContent = bottlenecksCard.querySelector('.metric-value');
        
        const nodeList = document.createElement('div');
        nodeList.className = 'node-list';
        
        data.bottlenecks.forEach(item => {
            const listItem = document.createElement('div');
            listItem.className = 'node-list-item';
            
            const nodeInfo = document.createElement('span');
            nodeInfo.textContent = `${item.node} (Betweenness: ${item.betweenness.toFixed(4)})`;
            
            const highlightBtn = document.createElement('button');
            highlightBtn.className = 'highlight-button';
            highlightBtn.textContent = 'Focus';
            highlightBtn.addEventListener('click', () => {
                if (network) {
                    // Clear previous highlights
                    clearNodeHighlights();
                    
                    // Highlight this node
                    highlightNodes([item.node]);
                    
                    // Focus on this node
                    network.focus(item.node, {
                        scale: 1.5,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            });
            
            listItem.appendChild(nodeInfo);
            listItem.appendChild(highlightBtn);
            nodeList.appendChild(listItem);
        });
        
        bottlenecksContent.appendChild(nodeList);
        container.appendChild(bottlenecksCard);
    }
    
    // Top intervention nodes
    if (data.top_intervention_nodes && data.top_intervention_nodes.length > 0) {
        const interventionCard = createMetricCard('Top Intervention Targets', '');
        const interventionContent = interventionCard.querySelector('.metric-value');
        
        const nodeList = document.createElement('div');
        nodeList.className = 'node-list';
        
        data.top_intervention_nodes.forEach(item => {
            const listItem = document.createElement('div');
            listItem.className = 'node-list-item';
            
            const nodeInfo = document.createElement('span');
            nodeInfo.textContent = `${item.node} (Impact: ${item.impact} nodes)`;
            
            const highlightBtn = document.createElement('button');
            highlightBtn.className = 'highlight-button';
            highlightBtn.textContent = 'Focus';
            highlightBtn.addEventListener('click', () => {
                if (network) {
                    // Clear previous highlights
                    clearNodeHighlights();
                    
                    // Highlight this node
                    highlightNodes([item.node]);
                    
                    // Focus on this node
                    network.focus(item.node, {
                        scale: 1.5,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            });
            
            listItem.appendChild(nodeInfo);
            listItem.appendChild(highlightBtn);
            nodeList.appendChild(listItem);
        });
        
        interventionContent.appendChild(nodeList);
        container.appendChild(interventionCard);
    }
}

/**
 * Render community analysis results
 */
function renderCommunityAnalysis(data, container) {
    // Community count
    if (data.count !== undefined) {
        const countCard = createMetricCard('Community Count', data.count);
        container.appendChild(countCard);
    }
    
    // Bridging nodes
    if (data.bridging_nodes && data.bridging_nodes.length > 0) {
        const bridgingCard = createMetricCard('Community Bridging Nodes', '');
        const bridgingContent = bridgingCard.querySelector('.metric-value');
        
        const description = document.createElement('p');
        description.textContent = `Nodes that connect different communities (top ${data.bridging_nodes.length}):`;
        bridgingContent.appendChild(description);
        
        const nodeList = document.createElement('div');
        nodeList.className = 'node-list';
        
        data.bridging_nodes.forEach(item => {
            const listItem = document.createElement('div');
            listItem.className = 'node-list-item';
            
            const nodeInfo = document.createElement('span');
            nodeInfo.textContent = `${item.node} (Connects ${item.bridge_count} communities)`;
            
            const highlightBtn = document.createElement('button');
            highlightBtn.className = 'highlight-button';
            highlightBtn.textContent = 'Focus';
            highlightBtn.addEventListener('click', () => {
                if (network) {
                    // Clear previous highlights
                    clearNodeHighlights();
                    
                    // Highlight this node
                    highlightNodes([item.node]);
                    
                    // Focus on this node
                    network.focus(item.node, {
                        scale: 1.5,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            });
            
            listItem.appendChild(nodeInfo);
            listItem.appendChild(highlightBtn);
            nodeList.appendChild(listItem);
        });
        
        bridgingContent.appendChild(nodeList);
        container.appendChild(bridgingCard);
    }
}

/**
 * Render topic clusters analysis results
 */
function renderTopicAnalysis(data, container) {
    // Topic cluster count
    const countCard = createMetricCard('Topic Statistics', '');
    const countContent = countCard.querySelector('.metric-value');
    
    const clusterCount = document.createElement('p');
    clusterCount.textContent = `Total topic clusters: ${data.cluster_count}`;
    countContent.appendChild(clusterCount);
    
    const nodesWithTopics = document.createElement('p');
    nodesWithTopics.textContent = `Nodes with identified topics: ${data.nodes_with_topics}`;
    countContent.appendChild(nodesWithTopics);
    
    const nodesWithoutTopics = document.createElement('p');
    nodesWithoutTopics.textContent = `Nodes without topics: ${data.nodes_without_topics}`;
    countContent.appendChild(nodesWithoutTopics);
    
    container.appendChild(countCard);
    
    // Topic clusters
    if (data.topic_clusters && data.topic_clusters.length > 0) {
        const topicClustersCard = createMetricCard('Topic Clusters', '');
        const topicClustersContent = topicClustersCard.querySelector('.metric-value');
        
        // Create a collapsible section for each topic cluster
        data.topic_clusters.forEach((cluster, index) => {
            const section = document.createElement('div');
            section.className = 'collapsible-section';
            
            const header = document.createElement('div');
            header.className = 'collapsible-header';
            header.textContent = `${cluster.topic} (${cluster.size} nodes)`;
            header.addEventListener('click', () => {
                header.classList.toggle('expanded');
                content.classList.toggle('expanded');
            });
            
            const content = document.createElement('div');
            content.className = 'collapsible-content';
            
            // Add nodes in this cluster
            const nodeCount = Math.min(cluster.nodes.length, 20); // Limit to 20 nodes for display
            const nodeText = document.createElement('p');
            nodeText.textContent = `Showing ${nodeCount} of ${cluster.nodes.length} nodes:`;
            content.appendChild(nodeText);
            
            // Create list of nodes
            const nodesList = document.createElement('ul');
            nodesList.style.marginTop = '5px';
            nodesList.style.paddingLeft = '20px';
            
            for (let i = 0; i < nodeCount; i++) {
                const listItem = document.createElement('li');
                listItem.textContent = cluster.nodes[i];
                nodesList.appendChild(listItem);
            }
            
            content.appendChild(nodesList);
            
            // Add a button to highlight all nodes in this topic cluster
            const highlightBtn = document.createElement('button');
            highlightBtn.className = 'highlight-button';
            highlightBtn.style.marginTop = '10px';
            highlightBtn.textContent = `Highlight "${cluster.topic}" Cluster`;
            highlightBtn.addEventListener('click', () => {
                if (network) {
                    // Clear previous highlights
                    clearNodeHighlights();
                    
                    // Highlight nodes in this cluster
                    highlightNodes(cluster.nodes);
                    
                    // Focus on these nodes
                    network.fit({
                        nodes: cluster.nodes,
                        animation: {
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }
                    });
                }
            });
            
            content.appendChild(highlightBtn);
            
            section.appendChild(header);
            section.appendChild(content);
            topicClustersContent.appendChild(section);
        });
        
        container.appendChild(topicClustersCard);
    }
}

/**
 * Create a metric card with header and content
 */
function createMetricCard(title, value) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    
    const header = document.createElement('div');
    header.className = 'metric-header';
    header.textContent = title;
    
    const content = document.createElement('div');
    content.className = 'metric-value';
    
    if (value !== '') {
        if (typeof value === 'object' && value !== null) {
            content.textContent = JSON.stringify(value, null, 2);
        } else {
            content.textContent = value;
        }
    }
    
    card.appendChild(header);
    card.appendChild(content);
    return card;
}

/**
 * Format a key string to be more readable
 */
function formatKey(key) {
    return key
        .replace(/_/g, ' ')
        .replace(/\b\w/g, l => l.toUpperCase());
}