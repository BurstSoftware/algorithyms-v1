import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import time

# Set page config
st.set_page_config(page_title="Algorithms Explorer", layout="wide")

# Sidebar navigation
st.sidebar.title("Algorithms Explorer")
category = st.sidebar.selectbox("Select Category", [
    "Sorting Algorithms", "Search Algorithms", "Graph Algorithms", 
    "Dynamic Programming Algorithms", "Greedy Algorithms", "Mathematical Algorithms",
    "Machine Learning Algorithms", "String Algorithms", "Brute Force and Exhaustive Methods",
    "Cryptographic Algorithms", "Data Compression Algorithms", "Computational Geometry Algorithms",
    "Optimization Algorithms", "Parallel and Distributed Algorithms", "Randomized Algorithms",
    "Pathfinding and Planning Algorithms", "Numerical Analysis Algorithms", "Game Theory Algorithms",
    "Image Processing Algorithms", "Network Flow Algorithms", "Approximation Algorithms",
    "Scheduling Algorithms", "Quantum Computing Algorithms", "Bioinformatics Algorithms",
    "Database Algorithms", "Robotics Algorithms", "Natural Language Processing (NLP) Algorithms",
    "Signal Processing Algorithms", "Online Algorithms", "Error-Correcting Code Algorithms",
    "Financial Algorithms", "Simulation Algorithms", "Hardware-Specific Algorithms",
    "Artificial Intelligence (AI) Planning Algorithms", "Data Mining Algorithms",
    "Computer Vision Algorithms", "Concurrency Control Algorithms", "Blockchain Algorithms",
    "Augmented Reality (AR) Algorithms", "Ethical AI Algorithms", "Fuzzy Logic Algorithms",
    "Statistical Algorithms", "Evolutionary Algorithms", "Recommendation Systems",
    "Internet of Things (IoT) Algorithms", "Gaming Algorithms", "Chaos Theory Algorithms",
    "Decision-Making Algorithms", "Audio Processing Algorithms", "Multi-Agent Systems",
    "Graph Drawing Algorithms", "Climate Modeling Algorithms", "Social Network Analysis Algorithms",
    "Synthetic Biology Algorithms", "Time Series Analysis Algorithms", "Load Balancing Algorithms",
    "Constraint Satisfaction Algorithms", "Text Mining Algorithms", "Cybersecurity Algorithms",
    "Virtual Reality (VR) Algorithms", "Graph Neural Network (GNN) Algorithms",
    "Operations Research Algorithms", "Pattern Recognition Algorithms",
    "Predictive Maintenance Algorithms", "Big Data Algorithms", "Autonomous Vehicle Algorithms",
    "Neuromorphic Computing Algorithms", "Swarm Robotics Algorithms", "Geospatial Algorithms",
    "Reinforcement Learning Algorithms", "Knowledge Representation Algorithms",
    "Energy Optimization Algorithms", "Quantum Machine Learning Algorithms",
    "Bioinformatics Workflow Algorithms", "Ethical Decision-Making Algorithms",
    "Supply Chain Optimization Algorithms", "Natural Computing Algorithms",
    "Semantic Web Algorithms", "Healthcare Algorithms", "Urban Planning Algorithms",
    "Sports Analytics Algorithms", "Cognitive Modeling Algorithms",
    "Human-Computer Interaction (HCI) Algorithms", "Satellite Data Processing Algorithms",
    "Synthetic Media Algorithms", "Adversarial Machine Learning Algorithms"
])

# Dictionary of algorithms and descriptions
algorithms = {
    "Sorting Algorithms": {
        "Bubble Sort": "Swaps adjacent elements iteratively.",
        "Merge Sort": "Divides, sorts, and merges sub-arrays.",
        "Quick Sort": "Partitions around a pivot recursively.",
        "Insertion Sort": "Builds a sorted list incrementally.",
        "Heap Sort": "Uses a heap to sort by extracting maxima."
    },
    "Search Algorithms": {
        "Binary Search": "Halves search space in sorted data.",
        "Linear Search (Brute Force Search)": "Checks all elements sequentially.",
        "Depth-First Search (DFS)": "Explores graph branches deeply.",
        "Breadth-First Search (BFS)": "Explores graph levels systematically.",
        "k-Nearest Neighbors (kNN) Search": "Finds k closest points by distance."
    },
    "Graph Algorithms": {
        "Dijkstra’s Algorithm": "Shortest path with non-negative weights.",
        "Bellman-Ford Algorithm": "Shortest path with negative weights.",
        "Kruskal’s Algorithm": "Minimum spanning tree via edge sorting.",
        "Prim’s Algorithm": "Grows a minimum spanning tree.",
        "Floyd-Warshall Algorithm": "All-pairs shortest paths."
    },
    # Add other categories similarly (abridged for brevity)
}

# Function to visualize Bubble Sort
def bubble_sort_visual(arr):
    steps = []
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
            steps.append(arr.copy())
    return steps

# Function to visualize kNN
def knn_visual():
    X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_clusters_per_class=1, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    return fig

# Main app logic
st.title("Algorithms Explorer")
st.write("Explore various algorithms across multiple domains with descriptions and visualizations.")

if category in algorithms:
    st.header(category)
    for algo, desc in algorithms[category].items():
        st.subheader(algo)
        st.write(f"**Description**: {desc}")
        
        # Add visualizations or demos for specific algorithms
        if algo == "Bubble Sort":
            st.write("**Visualization**: Watch Bubble Sort in action!")
            arr = [64, 34, 25, 12, 22, 11, 90]
            steps = bubble_sort_visual(arr.copy())
            step = st.slider("Step", 0, len(steps)-1, 0, key=f"bubble_{algo}")
            fig, ax = plt.subplots()
            ax.bar(range(len(steps[step])), steps[step], color='skyblue')
            ax.set_title("Bubble Sort Progress")
            st.pyplot(fig)
        
        elif algo == "k-Nearest Neighbors (kNN) Search":
            st.write("**Visualization**: kNN classification on synthetic data.")
            fig = knn_visual()
            st.pyplot(fig)
        
        # Placeholder for other algorithms
        else:
            st.write("Visualization or demo not yet implemented for this algorithm.")

else:
    st.header(category)
    st.write("This category is under development. Please check back later!")

# Footer
st.sidebar.write("Built with Streamlit | © 2025 Algorithms Explorer")
