import networkx as nx
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class StableHIVContactNetworkGenerator:
    def __init__(self, 
                 total_nodes=5389, 
                 largest_component_size=2387, 
                 start_date=datetime(1986, 1, 1), 
                 end_date=datetime(2004, 12, 31),
                 new_nodes_per_month=10,  # Limit new node additions
                 new_edges_per_month=5,   # Limit new edge additions
                 edge_rewire_prob=0.05,   # Probability of rewiring existing edges
                 seed=42):
        """
        Generate a stable, incrementally changing HIV contact tracing network
        
        Parameters:
        - new_nodes_per_month: Maximum number of new nodes added monthly
        - new_edges_per_month: Maximum number of new edges added monthly
        - edge_rewire_prob: Probability of rewiring existing edges
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.total_nodes = total_nodes
        self.largest_component_size = largest_component_size
        self.start_date = start_date
        self.end_date = end_date
        self.new_nodes_per_month = new_nodes_per_month
        self.new_edges_per_month = new_edges_per_month
        self.edge_rewire_prob = edge_rewire_prob
        
        # Generate the base graph
        self.base_graph = self._generate_initial_network()
        
        # Store monthly graphs
        self.monthly_graphs = self._generate_stable_monthly_snapshots()
    
    def _generate_initial_network(self):
        """
        Generate an initial stable network
        """
        # Create a scale-free graph base
        G = nx.barabasi_albert_graph(self.largest_component_size, 2)
        
        # Add node attributes
        for node in G.nodes():
            # Spread initial detection dates
            G.nodes[node]['detection_date'] = self.start_date + timedelta(
                days=random.randint(0, 180)
            )
            G.nodes[node]['gender'] = random.choice(['M', 'F'])
            G.nodes[node]['age'] = random.randint(15, 65)
        
        return G
    
    def _add_new_nodes(self, G, current_date):
        """
        Add a small number of new nodes to the graph
        """
        # Number of new nodes to add
        num_new_nodes = random.randint(1, self.new_nodes_per_month)
        
        for _ in range(num_new_nodes):
            # Add a new node connected to existing nodes
            new_node = G.number_of_nodes()
            
            # Connect to 1-2 existing nodes preferentially
            existing_nodes = list(G.nodes())
            connection_nodes = random.sample(
                existing_nodes, 
                min(2, len(existing_nodes))
            )
            
            # Add the new node and edges
            G.add_node(new_node)
            for connect_node in connection_nodes:
                G.add_edge(new_node, connect_node)
            
            # Add node attributes
            G.nodes[new_node]['detection_date'] = current_date
            G.nodes[new_node]['gender'] = random.choice(['M', 'F'])
            G.nodes[new_node]['age'] = random.randint(15, 65)
        
        return G
    
    def _modify_existing_edges(self, G):
        """
        Slightly modify existing edges to simulate network evolution
        """
        # Potential edge rewiring
        edges = list(G.edges())
        for edge in edges:
            if random.random() < self.edge_rewire_prob:
                # Remove the original edge
                G.remove_edge(*edge)
                
                # Find new potential connection
                possible_nodes = list(set(G.nodes()) - set(edge))
                new_target = random.choice(possible_nodes)
                
                # Add a new edge
                G.add_edge(edge[0], new_target)
        
        return G
    
    def _generate_stable_monthly_snapshots(self):
        """
        Create monthly graph snapshots with minimal, controlled changes
        """
        monthly_graphs = {}
        current_graph = self.base_graph.copy()
        current_date = self.start_date
        
        while current_date <= self.end_date:
            # Add a small number of new nodes
            current_graph = self._add_new_nodes(current_graph, current_date)
            
            # Slightly modify existing edges
            current_graph = self._modify_existing_edges(current_graph)
            
            # Ensure graph doesn't grow beyond total nodes
            if current_graph.number_of_nodes() > self.total_nodes:
                # Remove oldest detected nodes if necessary
                oldest_nodes = sorted(
                    current_graph.nodes(), 
                    key=lambda n: current_graph.nodes[n]['detection_date']
                )[:current_graph.number_of_nodes() - self.total_nodes]
                current_graph.remove_nodes_from(oldest_nodes)
            
            # Store the graph if it has enough nodes
            if current_graph.number_of_nodes() >= 500:
                monthly_graphs[current_date] = current_graph.copy()
            
            # Move to next month
            current_date += timedelta(days=30)
        
        return monthly_graphs
    
    def export_monthly_graphs(self, output_dir='stable_hiv_contact_network'):
        """
        Export monthly graphs to CSV files
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for date, graph in self.monthly_graphs.items():
            # Prepare edge list
            edges_df = pd.DataFrame(list(graph.edges()), columns=['source', 'target'])
            
            # Add node attributes
            node_attrs = []
            for node in graph.nodes():
                node_attrs.append({
                    'node': node,
                    'detection_date': graph.nodes[node]['detection_date'],
                    'gender': graph.nodes[node]['gender'],
                    'age': graph.nodes[node]['age']
                })
            nodes_df = pd.DataFrame(node_attrs)
            
            # Save to CSV
            filename = f"{output_dir}/{date.strftime('%Y-%m')}_graph"
            edges_df.to_csv(filename + '_edges.csv', index=False)
            nodes_df.to_csv(filename + '_nodes.csv', index=False)
        
        return self.monthly_graphs

# Generate the network
generator = StableHIVContactNetworkGenerator()

# Export monthly graphs
monthly_graphs = generator.export_monthly_graphs()

# Print graph evolution
print(f"Total monthly graphs: {len(monthly_graphs)}")
for date, graph in monthly_graphs.items():
    print(f"Date: {date}, Nodes: {len(graph)}, Edges: {len(graph.edges())}")