import os
import networkx as nx
import pandas as pd
import numpy as np

def load_monthly_graphs(input_dir='stable_hiv_contact_network'):
    graph_count = 0
    monthly_graphs = {}
    for file in os.listdir(input_dir):
        if file.endswith("_edges.csv"):
            date_str = file.split("_graph_edges.csv")[0]
            edges_path = os.path.join(input_dir, file)
            nodes_path = os.path.join(input_dir, f"{date_str}_graph_nodes.csv")
            
            edges_df = pd.read_csv(edges_path)
            nodes_df = pd.read_csv(nodes_path)
            
            G = nx.from_pandas_edgelist(edges_df, 'source', 'target')
            
            nodes_in_graph = set(G.nodes())
            nodes_df_filtered = nodes_df[nodes_df['node'].isin(nodes_in_graph)]
            
            for _, row in nodes_df_filtered.iterrows():
                node = row['node']
                G.nodes[node]['detection_date'] = pd.to_datetime(row['detection_date'])
                G.nodes[node]['gender'] = row['gender']
                G.nodes[node]['age'] = row['age']
            
            monthly_graphs[pd.to_datetime(date_str)] = G
        graph_count += 1
        if graph_count == 10:
            return monthly_graphs
    
    return monthly_graphs

monthly_graphs = load_monthly_graphs()


def compute_shifted_laplacian(G, shift=0.01):
    L = nx.laplacian_matrix(G).astype(float)  # laplacian
    I = np.eye(len(G)) # identity
    return L + shift * I


from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

def spectral_clustering(graphs, k, l, recompute_step):
    cluster_results = []
    previous_centroids = None
    previous_eigvecs = None
    previous_eigvals = None
    
    for i, (date, G) in enumerate(graphs.items()):
        print(f"Processing graph for {date.strftime('%Y-%m')}...")
        
        # compute shifted Laplacian
        L_hat = compute_shifted_laplacian(G)
        
        # Recompute or approximate eigenvectors
        if i % recompute_step == 0 or previous_eigvecs is None:
            # recompute evs and ews if it's time or no previous eigenvectors
            eigvals, eigvecs = eigsh(L_hat, k=l, which='LM')
        else:
            # hmmm
            eigvals, eigvecs = eigsh(L_hat, k=l, which='LM', v0=previous_eigvecs)
            eigvecs = update_eigenvectors(previous_eigvecs, eigvecs, l)
        
        # normalize (rows of evectors)
        V_k = eigvecs[:, :k]
        row_norms = np.linalg.norm(V_k, axis=1, keepdims=True)
        V_k_normalized = V_k / row_norms
        
        # k-means clustering with appropriate initialization
        if previous_centroids is None:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10)
        else:
            kmeans = KMeans(n_clusters=k, init=previous_centroids, n_init=10)
        
        labels = kmeans.fit_predict(V_k_normalized)
        previous_centroids = kmeans.cluster_centers_  # update centroids for next iteration
        
        cluster_results.append((date, labels))
        
        previous_eigvecs = eigvecs
        previous_eigvals = eigvals
    
    return cluster_results

def update_eigenvectors(previous_eigvecs, current_eigvecs, l):
    # placeholder for the rank-l update that I can't figure out
    return np.hstack((previous_eigvecs[:, :l], current_eigvecs[:, :l]))

k = 3  # num clusters
l = 5  # approx rank
recompute_step = 3  # Recompute eigen-decomposition every 3 graphs
clusters = spectral_clustering(monthly_graphs, k, l, recompute_step)

import matplotlib.pyplot as plt

def plot_clusters(graphs, clusters):
    """
    Visualize the clusters on the graphs.
    """
    for (date, G), (_, labels) in zip(graphs.items(), clusters):
        plt.figure(figsize=(8, 8))
        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(
            G, pos, node_size=50, node_color=labels, cmap=plt.cm.tab10
        )
        
        nx.draw_networkx_edges(G, pos, alpha=0.2)
        
        plt.title(f"Graph for {date.strftime('%Y-%m')}")
        plt.show()

plot_clusters(monthly_graphs, clusters)
