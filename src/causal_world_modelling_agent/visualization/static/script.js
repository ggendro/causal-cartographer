
// Global variables
let network = null;
let graphData = null;
let nodesDataset = null;
let edgesDataset = null;
let selectedElement = null;
let physicsEnabled = true;
let highlightedNodes = new Set();
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
 * Clear all node highlights
 */
function clearNodeHighlights() {
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
}

/**
 * Highlight specific nodes
 */
function highlightNodes(nodeIds) {
    if (nodesDataset) {
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
    }
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