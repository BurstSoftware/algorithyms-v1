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

# Dictionary of algorithms and descriptions (complete list)
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
    "Dynamic Programming Algorithms": {
        "Fibonacci Sequence (DP)": "Memoizes Fibonacci numbers.",
        "Knapsack Problem (0/1)": "Optimizes item selection.",
        "Longest Common Subsequence (LCS)": "Finds common subsequences."
    },
    "Greedy Algorithms": {
        "Huffman Coding": "Optimal prefix codes for compression.",
        "Activity Selection Problem": "Maximizes non-overlapping activities."
    },
    "Mathematical Algorithms": {
        "Euclidean Algorithm": "Computes GCD via division.",
        "Sieve of Eratosthenes": "Generates prime numbers.",
        "Fast Fourier Transform (FFT)": "Efficient Fourier transform."
    },
    "Machine Learning Algorithms": {
        "Linear Regression": "Fits a linear model to data.",
        "k-Nearest Neighbors (kNN)": "Predicts via k nearest points.",
        "K-Means Clustering": "Partitions data into k clusters.",
        "Decision Tree": "Splits data for decisions.",
        "Gradient Descent": "Minimizes loss iteratively."
    },
    "String Algorithms": {
        "Knuth-Morris-Pratt (KMP)": "Substring search with prefix table.",
        "Rabin-Karp Algorithm": "Hash-based substring search.",
        "Boyer-Moore Algorithm": "Skips text for efficiency.",
        "Brute Force String Matching": "Checks all positions."
    },
    "Brute Force and Exhaustive Methods": {
        "Brute Force Search (General)": "Tries all possibilities.",
        "Brute Force Traveling Salesman Problem (TSP)": "Tests all routes.",
        "Brute Force Knapsack": "Evaluates all combinations.",
        "Exhaustive Search": "Explores all solutions systematically."
    },
    "Cryptographic Algorithms": {
        "SHA-256": "Secure hash function.",
        "RSA Algorithm": "Public-key encryption via primes.",
        "AES (Advanced Encryption Standard)": "Symmetric encryption.",
        "MD5": "Hash function (now insecure)."
    },
    "Data Compression Algorithms": {
        "Huffman Coding": "Variable-length compression (also Greedy).",
        "Lempel-Ziv-Welch (LZW)": "Dictionary-based compression.",
        "Run-Length Encoding (RLE)": "Compresses repeated sequences.",
        "Burrows-Wheeler Transform": "Reorders data for compression."
    },
    "Computational Geometry Algorithms": {
        "Graham’s Scan": "Computes convex hull.",
        "Closest Pair of Points": "Finds nearest pair in a plane.",
        "Line Intersection Algorithm": "Detects line segment intersections.",
        "Voronoi Diagram Algorithm": "Partitions plane by distance."
    },
    "Optimization Algorithms": {
        "Simulated Annealing": "Probabilistic optimization.",
        "Genetic Algorithm": "Evolves solutions via selection.",
        "Hill Climbing": "Locally improves solutions.",
        "Linear Programming (Simplex Method)": "Optimizes linear functions."
    },
    "Parallel and Distributed Algorithms": {
        "MapReduce": "Parallel processing of large datasets.",
        "Parallel Prefix Sum (Scan)": "Computes sums in parallel.",
        "Bulk Synchronous Parallel (BSP)": "Synchronizes parallel tasks.",
        "Distributed Hash Table (DHT)": "Decentralized key-value store."
    },
    "Randomized Algorithms": {
        "Monte Carlo Method": "Random sampling for estimation.",
        "Randomized Quick Sort": "Random pivot for efficiency.",
        "Las Vegas Algorithm (e.g., Randomized Min-Cut)": "Ensures correctness randomly.",
        "Reservoir Sampling": "Randomly selects k items from a stream."
    },
    "Pathfinding and Planning Algorithms": {
        "A* Search Algorithm": "Heuristic-guided pathfinding.",
        "D* (Dynamic A*)": "Adapts A* dynamically.",
        "Rapidly-exploring Random Tree (RRT)": "Explores for motion planning.",
        "Greedy Best-First Search": "Prioritizes goal proximity."
    },
    "Numerical Analysis Algorithms": {
        "Newton-Raphson Method": "Finds roots iteratively.",
        "Gaussian Elimination": "Solves linear equations.",
        "Runge-Kutta Method (RK4)": "Solves differential equations.",
        "Bisection Method": "Finds roots by halving intervals."
    },
    "Game Theory Algorithms": {
        "Minimax Algorithm": "Optimizes two-player game moves.",
        "Alpha-Beta Pruning": "Enhances Minimax efficiency.",
        "Nash Equilibrium Solver": "Finds stable strategies.",
        "Monte Carlo Tree Search (MCTS)": "Explores game trees randomly."
    },
    "Image Processing Algorithms": {
        "Canny Edge Detection": "Identifies image edges.",
        "Sobel Operator": "Computes gradients for edges.",
        "Hough Transform": "Detects shapes in images.",
        "Gaussian Blur": "Smooths images with a filter."
    },
    "Network Flow Algorithms": {
        "Ford-Fulkerson Algorithm": "Computes maximum flow.",
        "Edmonds-Karp Algorithm": "Enhances Ford-Fulkerson with BFS.",
        "Dinic’s Algorithm": "Efficient flow with level graphs.",
        "Push-Relabel Algorithm": "Optimizes flow with preflow."
    },
    "Approximation Algorithms": {
        "Greedy Set Cover": "Approximates set cover.",
        "Christofides Algorithm": "Approximates TSP (3/2 bound).",
        "k-Means++": "Improves k-Means initialization.",
        "Vertex Cover Approximation": "Approximates vertex cover."
    },
    "Scheduling Algorithms": {
        "First-Come, First-Served (FCFS)": "Processes in arrival order.",
        "Shortest Job First (SJF)": "Prioritizes shortest tasks.",
        "Round Robin (RR)": "Allocates time slices cyclically.",
        "Earliest Deadline First (EDF)": "Schedules by deadlines."
    },
    "Quantum Computing Algorithms": {
        "Shor’s Algorithm": "Factors integers exponentially faster.",
        "Grover’s Algorithm": "Searches quadratically faster.",
        "Quantum Fourier Transform (QFT)": "Basis for quantum algorithms.",
        "Deutsch-Jozsa Algorithm": "Determines function balance."
    },
    "Bioinformatics Algorithms": {
        "Needleman-Wunsch Algorithm": "Global sequence alignment.",
        "Smith-Waterman Algorithm": "Local sequence alignment.",
        "BLAST (Basic Local Alignment Search Tool)": "Searches biological sequences.",
        "Hidden Markov Model (HMM) Viterbi Algorithm": "Finds likely state sequences."
    },
    "Database Algorithms": {
        "B-Tree Insertion/Search": "Balances tree for efficient queries.",
        "Hash Join": "Joins tables using hashing.",
        "Quicksort for Sorting Indexes": "Sorts database indexes.",
        "Bloom Filter": "Probabilistic membership testing."
    },
    "Robotics Algorithms": {
        "SLAM (Simultaneous Localization and Mapping)": "Maps and locates in unknown environments.",
        "PID Control Algorithm": "Adjusts motion via feedback.",
        "Particle Filter": "Estimates position probabilistically.",
        "RANSAC (Random Sample Consensus)": "Fits models to noisy data."
    },
    "Natural Language Processing (NLP) Algorithms": {
        "TF-IDF (Term Frequency-Inverse Document Frequency)": "Weights word importance.",
        "Word2Vec": "Embeds words in vector space.",
        "N-Gram Model": "Predicts next word based on prior words.",
        "Levenshtein Distance": "Measures string edit distance."
    },
    "Signal Processing Algorithms": {
        "Discrete Cosine Transform (DCT)": "Compresses signals (e.g., JPEG).",
        "Wavelet Transform": "Analyzes signals at multiple scales.",
        "Kalman Filter": "Estimates state from noisy measurements.",
        "Goertzel Algorithm": "Detects specific frequencies."
    },
    "Online Algorithms": {
        "Online Greedy Algorithm": "Makes decisions without future knowledge.",
        "LRU (Least Recently Used) Cache": "Evicts least recently used items.",
        "Competitive Paging Algorithm": "Manages memory with limited foresight.",
        "Online k-Server Problem": "Minimizes server movement costs."
    },
    "Error-Correcting Code Algorithms": {
        "Hamming Code": "Detects and corrects single-bit errors.",
        "Reed-Solomon Code": "Corrects multiple errors in data.",
        "CRC (Cyclic Redundancy Check)": "Detects data corruption.",
        "LDPC (Low-Density Parity-Check)": "Efficient error correction."
    },
    "Financial Algorithms": {
        "Black-Scholes Option Pricing": "Calculates option prices.",
        "Monte Carlo Simulation for Finance": "Estimates financial outcomes.",
        "Fibonacci Retracement Algorithm": "Identifies stock price levels.",
        "Moving Average Convergence Divergence (MACD)": "Analyzes market trends."
    },
    "Simulation Algorithms": {
        "Gillespie Algorithm": "Simulates stochastic chemical reactions.",
        "Metropolis-Hastings Algorithm": "Samples from probability distributions.",
        "Agent-Based Modeling": "Simulates individual agent interactions.",
        "Event-Driven Simulation": "Processes discrete events in time."
    },
    "Hardware-Specific Algorithms": {
        "Bitonic Sort": "Sorts in parallel hardware (e.g., GPUs).",
        "CORDIC Algorithm": "Computes trigonometric functions in hardware.",
        "Booth’s Multiplication Algorithm": "Efficient binary multiplication.",
        "Strassen’s Matrix Multiplication": "Optimizes matrix operations."
    },
    "Artificial Intelligence (AI) Planning Algorithms": {
        "STRIPS Planning": "Plans actions to achieve goals.",
        "PDDL (Planning Domain Definition Language) Solver": "Solves planning problems.",
        "Hierarchical Task Network (HTN)": "Decomposes tasks hierarchically.",
        "Goal Stack Planning": "Breaks goals into sub-goals."
    },
    "Data Mining Algorithms": {
        "Apriori Algorithm": "Finds frequent itemsets in transactions.",
        "FP-Growth (Frequent Pattern Growth)": "Mines patterns without candidate generation.",
        "DBSCAN (Density-Based Spatial Clustering)": "Clusters based on density.",
        "PageRank": "Ranks web pages by importance."
    },
    "Computer Vision Algorithms": {
        "SIFT (Scale-Invariant Feature Transform)": "Detects and describes features.",
        "HOG (Histogram of Oriented Gradients)": "Extracts features for object detection.",
        "Optical Flow (Lucas-Kanade)": "Tracks motion in video.",
        "Viola-Jones Face Detection": "Detects faces using Haar features."
    },
    "Concurrency Control Algorithms": {
        "Two-Phase Locking (2PL)": "Ensures serializability in databases.",
        "Timestamp Ordering": "Orders transactions by timestamps.",
        "Readers-Writers Lock": "Manages concurrent access.",
        "Peterson’s Algorithm": "Ensures mutual exclusion in two processes."
    },
    "Blockchain Algorithms": {
        "Proof of Work (PoW)": "Secures blockchain via computational effort.",
        "Proof of Stake (PoS)": "Validates transactions based on stake.",
        "SHA-256 (in Blockchain Context)": "Hashes blocks (also Cryptographic).",
        "Merkle Tree Construction": "Efficiently verifies data integrity."
    },
    "Augmented Reality (AR) Algorithms": {
        "Marker-Based Tracking": "Detects predefined markers for AR overlay.",
        "Feature Matching (e.g., ORB)": "Aligns real-world features with digital content.",
        "Pose Estimation": "Determines object position/orientation in 3D.",
        "Depth Estimation": "Infers depth from stereo or sensor data."
    },
    "Ethical AI Algorithms": {
        "Fairness-Aware Classification": "Reduces bias in predictions.",
        "Explainable AI (e.g., LIME)": "Interprets model decisions.",
        "Differential Privacy": "Protects data privacy in analysis.",
        "Adversarial Debiasing": "Mitigates bias via adversarial training."
    },
    "Fuzzy Logic Algorithms": {
        "Fuzzy C-Means Clustering": "Clusters with soft memberships.",
        "Mamdani Fuzzy Inference": "Models decisions with fuzzy rules.",
        "Sugeno Fuzzy Inference": "Computes crisp outputs from fuzzy inputs.",
        "Defuzzification (Centroid Method)": "Converts fuzzy outputs to crisp values."
    },
    "Statistical Algorithms": {
        "Bayesian Inference": "Updates probabilities with new data.",
        "Principal Component Analysis (PCA)": "Reduces dimensionality.",
        "Expectation-Maximization (EM)": "Estimates parameters in latent models.",
        "Markov Chain Monte Carlo (MCMC)": "Samples from complex distributions."
    },
    "Evolutionary Algorithms": {
        "Genetic Algorithm": "Evolves solutions (also Optimization).",
        "Differential Evolution": "Optimizes via population differences.",
        "Particle Swarm Optimization (PSO)": "Mimics social behavior for optimization.",
        "Ant Colony Optimization (ACO)": "Solves paths via pheromone trails."
    },
    "Recommendation Systems": {
        "Collaborative Filtering": "Recommends based on user similarities.",
        "Content-Based Filtering": "Recommends via item features.",
        "Matrix Factorization (e.g., SVD)": "Decomposes user-item matrices.",
        "Hybrid Recommendation Algorithm": "Combines collaborative and content-based methods."
    },
    "Internet of Things (IoT) Algorithms": {
        "TinyML Inference": "Runs ML models on resource-constrained devices.",
        "Data Aggregation (e.g., Cluster-Based)": "Reduces IoT network traffic.",
        "LEACH (Low-Energy Adaptive Clustering Hierarchy)": "Optimizes energy in sensor networks.",
        "MQTT Message Routing": "Efficiently routes IoT messages."
    },
    "Gaming Algorithms": {
        "Pathfinding (A* in Games)": "Navigates game characters (also Pathfinding).",
        "Procedural Content Generation (PCG)": "Creates game levels dynamically.",
        "Behavior Trees": "Controls AI character actions.",
        "Flocking Algorithm (Boids)": "Simulates group movement in games."
    },
    "Chaos Theory Algorithms": {
        "Logistic Map": "Models chaotic population dynamics.",
        "Lyapunov Exponent Calculation": "Measures chaos sensitivity.",
        "Fractal Generation (e.g., Mandelbrot Set)": "Visualizes chaotic patterns.",
        "Strange Attractor Simulation": "Simulates chaotic system trajectories."
    },
    "Decision-Making Algorithms": {
        "Analytic Hierarchy Process (AHP)": "Prioritizes decisions with criteria.",
        "TOPSIS (Technique for Order Preference)": "Ranks alternatives.",
        "Decision Table Algorithm": "Evaluates rules for decisions.",
        "Multi-Attribute Utility Theory (MAUT)": "Combines preferences for choices."
    },
    "Audio Processing Algorithms": {
        "Pitch Detection Algorithm (e.g., Autocorrelation)": "Identifies audio pitch.",
        "MFCC (Mel-Frequency Cepstral Coefficients)": "Extracts audio features.",
        "Echo Cancellation": "Removes echo from audio signals.",
        "Dynamic Range Compression": "Adjusts audio volume levels."
    },
    "Multi-Agent Systems": {
        "Consensus Algorithm": "Aligns agent states (e.g., in robotics).",
        "Auction Algorithm": "Allocates tasks among agents.",
        "Swarm Intelligence (e.g., PSO)": "Coordinates agents (also Evolutionary).",
        "Distributed Constraint Optimization (DCOP)": "Solves multi-agent problems."
    },
    "Graph Drawing Algorithms": {
        "Force-Directed Graph Drawing": "Positions nodes aesthetically.",
        "Hierarchical Graph Layout": "Arranges nodes in layers.",
        "Spring Embedder Algorithm": "Balances node distances.",
        "Planar Graph Drawing (e.g., Fáry’s Theorem)": "Draws without edge crossings."
    },
    "Climate Modeling Algorithms": {
        "Finite Difference Method (FDM)": "Solves climate differential equations.",
        "Global Circulation Model (GCM)": "Simulates atmospheric dynamics.",
        "Carbon Cycle Model": "Tracks carbon flow in ecosystems.",
        "Ensemble Forecasting": "Predicts climate with multiple models."
    },
    "Social Network Analysis Algorithms": {
        "Centrality Measures (e.g., Betweenness)": "Identifies key nodes in networks.",
        "Community Detection (e.g., Louvain)": "Groups nodes into communities.",
        "Link Prediction": "Predicts future connections in networks.",
        "Girvan-Newman Algorithm": "Detects communities via edge removal."
    },
    "Synthetic Biology Algorithms": {
        "Gibson Assembly Algorithm": "Designs DNA fragment assembly.",
        "Codon Optimization": "Adjusts codons for gene expression.",
        "SBOL (Synthetic Biology Open Language) Parser": "Processes synthetic designs.",
        "Flux Balance Analysis (FBA)": "Models metabolic networks."
    },
    "Time Series Analysis Algorithms": {
        "ARIMA (AutoRegressive Integrated Moving Average)": "Forecasts time series.",
        "Holt-Winters Exponential Smoothing": "Predicts with trends and seasonality.",
        "Dynamic Time Warping (DTW)": "Measures time series similarity.",
        "Seasonal Decomposition": "Separates trends and seasonality."
    },
    "Load Balancing Algorithms": {
        "Round-Robin Load Balancing": "Distributes tasks cyclically.",
        "Least Connections": "Assigns tasks to least busy servers.",
        "Consistent Hashing": "Maps tasks to servers with minimal reassignment.",
        "Weighted Round-Robin": "Prioritizes servers by capacity."
    },
    "Constraint Satisfaction Algorithms": {
        "Backtracking Search": "Solves CSPs by exploring possibilities.",
        "Constraint Propagation (AC-3)": "Reduces variable domains.",
        "Min-Conflicts Algorithm": "Solves CSPs via local adjustments.",
        "DPLL Algorithm": "Solves satisfiability problems efficiently."
    },
    "Text Mining Algorithms": {
        "Latent Dirichlet Allocation (LDA)": "Extracts topics from text.",
        "Sentiment Analysis (e.g., VADER)": "Determines text sentiment.",
        "Named Entity Recognition (NER)": "Identifies entities in text.",
        "TextRank": "Ranks text elements (e.g., keywords)."
    },
    "Cybersecurity Algorithms": {
        "Intrusion Detection (e.g., Anomaly-Based)": "Identifies unusual network activity.",
        "Rainbow Table Attack": "Cracks hashed passwords.",
        "Elliptic Curve Cryptography (ECC)": "Efficient public-key encryption.",
        "Zero-Knowledge Proof (e.g., zk-SNARK)": "Proves knowledge without revealing it."
    },
    "Virtual Reality (VR) Algorithms": {
        "Head Tracking (e.g., Kalman-Based)": "Tracks user head movement.",
        "Ray Casting": "Renders VR scenes by tracing rays.",
        "Collision Detection (e.g., Bounding Volume)": "Detects object interactions in VR.",
        "Foveated Rendering": "Optimizes rendering based on eye focus."
    },
    "Graph Neural Network (GNN) Algorithms": {
        "Graph Convolutional Network (GCN)": "Learns node features in graphs.",
        "Graph Attention Network (GAT)": "Weights neighbor importance.",
        "Message Passing Neural Network (MPNN)": "Propagates information in graphs.",
        "GraphSAGE": "Samples neighbors for scalable GNN learning."
    },
    "Operations Research Algorithms": {
        "Branch and Bound": "Optimizes combinatorial problems.",
        "Cutting Plane Method": "Solves integer programming problems.",
        "Hungarian Algorithm": "Solves assignment problems.",
        "Queueing Theory (e.g., M/M/1 Model)": "Analyzes waiting lines."
    },
    "Pattern Recognition Algorithms": {
        "Support Vector Machine (SVM)": "Classifies data with hyperplanes.",
        "Hidden Markov Model (HMM)": "Models sequential patterns.",
        "Template Matching": "Identifies patterns in signals or images.",
        "k-Nearest Neighbors (kNN) for Recognition": "Classifies patterns (also ML)."
    },
    "Predictive Maintenance Algorithms": {
        "Anomaly Detection (e.g., Isolation Forest)": "Identifies equipment faults.",
        "Survival Analysis (e.g., Kaplan-Meier)": "Predicts time to failure.",
        "Condition-Based Monitoring": "Analyzes sensor data for maintenance.",
        "Prognostic Health Management (PHM)": "Forecasts component lifespan."
    },
    "Big Data Algorithms": {
        "Apache Spark RDD Operations": "Processes large datasets in memory.",
        "Hadoop MapReduce": "Distributes data processing (also Parallel).",
        "Locality-Sensitive Hashing (LSH)": "Approximates nearest neighbors in big data.",
        "HyperLogLog": "Estimates cardinality in massive datasets."
    },
    "Autonomous Vehicle Algorithms": {
        "Lane Detection (e.g., Hough-Based)": "Identifies road lanes.",
        "Obstacle Avoidance (e.g., Potential Fields)": "Navigates around obstacles.",
        "Sensor Fusion (e.g., Kalman Filter)": "Combines sensor data (also Signal Processing).",
        "Behavioral Cloning": "Learns driving from human examples."
    },
    "Neuromorphic Computing Algorithms": {
        "Spiking Neural Network (SNN) Training": "Mimics biological neurons.",
        "Hebbian Learning": "Strengthens connections based on activity.",
        "Event-Driven Processing": "Processes only active neuron events.",
        "Reservoir Computing": "Uses fixed recurrent networks for computation."
    },
    "Swarm Robotics Algorithms": {
        "Distributed Formation Control": "Aligns robots in patterns.",
        "Foraging Algorithm": "Coordinates resource collection.",
        "Swarm Gradient Descent": "Optimizes collectively via gradients.",
        "Stigmergy-Based Coordination": "Uses environmental cues for cooperation."
    },
    "Geospatial Algorithms": {
        "R-Tree Indexing": "Indexes spatial data efficiently.",
        "Delaunay Triangulation": "Connects points in a plane.",
        "Geohash Encoding": "Encodes locations into strings.",
        "Kriging": "Interpolates spatial data statistically."
    },
    "Reinforcement Learning Algorithms": {
        "Q-Learning": "Learns optimal actions via rewards.",
        "Deep Q-Network (DQN)": "Combines Q-learning with neural networks.",
        "Policy Gradient Method": "Optimizes action policies directly.",
        "SARSA (State-Action-Reward-State-Action)": "Updates based on next action."
    },
    "Knowledge Representation Algorithms": {
        "Description Logic Reasoning": "Infers from ontologies.",
        "Semantic Web Query (e.g., SPARQL)": "Queries structured knowledge.",
        "Frame-Based Reasoning": "Organizes knowledge in frames.",
        "Rule-Based Inference (e.g., Rete Algorithm)": "Applies rules to facts."
    },
    "Energy Optimization Algorithms": {
        "Demand Response Optimization": "Balances energy load.",
        "Dynamic Voltage and Frequency Scaling (DVFS)": "Adjusts power usage.",
        "Smart Grid Scheduling": "Optimizes energy distribution.",
        "Battery Management Algorithm": "Maximizes battery life."
    },
    "Quantum Machine Learning Algorithms": {
        "Quantum Support Vector Machine (QSVM)": "Classifies with quantum kernels.",
        "Variational Quantum Eigensolver (VQE)": "Optimizes quantum systems.",
        "Quantum Neural Network (QNN)": "Emulates neural networks on quantum hardware.",
        "Quantum k-Means Clustering": "Clusters data using quantum states."
    },
    "Bioinformatics Workflow Algorithms": {
        "Pipeline for RNA-Seq Analysis": "Processes gene expression data.",
        "Variant Calling (e.g., GATK)": "Identifies genetic variants.",
        "Metagenomics Classification": "Analyzes microbial communities.",
        "Phylogenetic Tree Construction": "Infers evolutionary relationships."
    },
    "Ethical Decision-Making Algorithms": {
        "Value Alignment Algorithm": "Aligns AI with human values.",
        "Moral Dilemma Resolver (e.g., Trolley Problem Models)": "Evaluates ethical choices.",
        "Bias Audit Algorithm": "Detects and mitigates biases.",
        "Transparency Scoring": "Assesses decision-making clarity."
    },
    "Supply Chain Optimization Algorithms": {
        "Vehicle Routing Problem (VRP) Solver": "Optimizes delivery routes.",
        "Inventory Management (EOQ Model)": "Balances stock levels.",
        "Supply Chain Simulation": "Models logistics dynamics.",
        "Linear Programming for Allocation": "Allocates resources efficiently."
    },
    "Natural Computing Algorithms": {
        "DNA Computing": "Solves problems using DNA molecules.",
        "Membrane Computing": "Mimics cellular processes for computation.",
        "Artificial Immune System (AIS)": "Models immune responses for optimization.",
        "Quantum-Inspired Annealing": "Uses quantum-like methods for optimization."
    },
    "Semantic Web Algorithms": {
        "RDF Triple Matching": "Queries semantic web data.",
        "Ontology Alignment": "Maps between knowledge bases.",
        "Reasoning with OWL (e.g., Pellet)": "Infers from ontologies.",
        "Linked Data Integration": "Combines distributed semantic data."
    },
    "Healthcare Algorithms": {
        "Medical Image Segmentation (e.g., U-Net)": "Segments organs in scans.",
        "Disease Prediction (e.g., Logistic Regression)": "Predicts health risks.",
        "Clinical Decision Support (CDSS)": "Assists medical decisions.",
        "Electronic Health Record (EHR) Clustering": "Groups patient records."
    },
    "Urban Planning Algorithms": {
        "Traffic Flow Optimization": "Minimizes congestion in cities.",
        "Land Use Allocation": "Optimizes zoning with constraints.",
        "Urban Growth Simulation": "Models city expansion.",
        "Public Transit Routing": "Designs efficient transit networks."
    },
    "Sports Analytics Algorithms": {
        "Player Performance Prediction": "Forecasts athlete stats.",
        "Game Strategy Optimization": "Plans team tactics.",
        "Injury Risk Assessment": "Predicts athlete injury likelihood.",
        "Expected Goals (xG) Model": "Estimates soccer scoring probability."
    },
    "Cognitive Modeling Algorithms": {
        "ACT-R (Adaptive Control of Thought-Rational)": "Simulates human cognition.",
        "SOAR Cognitive Architecture": "Models decision-making processes.",
        "Bayesian Cognitive Modeling": "Infers mental processes.",
        "Neural-Symbolic Integration": "Combines neural and symbolic reasoning."
    },
    "Human-Computer Interaction (HCI) Algorithms": {
        "Eye-Tracking Calibration": "Aligns gaze with interface.",
        "Gesture Recognition": "Interprets user movements.",
        "Adaptive User Interface (AUI)": "Customizes interfaces dynamically.",
        "Fitts’ Law Prediction": "Models pointing task efficiency."
    },
    "Satellite Data Processing Algorithms": {
        "SAR (Synthetic Aperture Radar) Processing": "Enhances radar images.",
        "NDVI (Normalized Difference Vegetation Index)": "Assesses vegetation health.",
        "Change Detection": "Identifies landscape changes.",
        "Atmospheric Correction": "Removes atmospheric distortions."
    },
    "Synthetic Media Algorithms": {
        "Deepfake Generation (e.g., GAN-Based)": "Creates realistic fake media.",
        "Style Transfer": "Applies artistic styles to media.",
        "Text-to-Image Synthesis (e.g., DALL-E)": "Generates images from text.",
        "Voice Synthesis (e.g., WaveNet)": "Produces lifelike audio."
    },
    "Adversarial Machine Learning Algorithms": {
        "Adversarial Attack (e.g., FGSM)": "Crafts inputs to fool models.",
        "Adversarial Defense (e.g., Robust Training)": "Strengthens models against attacks.",
        "GAN (Generative Adversarial Network)": "Trains generator and discriminator.",
        "Adversarial Example Detection": "Identifies malicious inputs."
    }
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
        
        elif algo == "k-Nearest Neighbors (kNN) Search" or algo == "k-Nearest Neighbors (kNN)":
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
