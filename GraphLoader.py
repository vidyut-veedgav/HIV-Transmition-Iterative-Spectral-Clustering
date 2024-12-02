import pandas as pd
import networkx as nx

class GraphLoader:
    """
    Stores edgelist data in a networkx graph representation
    """
    def __init__(self, filepath) -> None:
        data = pd.read_csv(filepath)
        self.graph = nx.from_pandas_edgelist(data)