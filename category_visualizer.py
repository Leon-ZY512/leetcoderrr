import networkx as nx
import matplotlib.pyplot as plt
from category_graph import CategoryGraph

class CategoryVisualizer:
    def __init__(self, graph_data):
        self.graph_data = graph_data

    def plot_graph(self):
        # Create a new directed graph
        G = nx.DiGraph()
        
        # Add edges with weights
        for source, targets in self.graph_data.items():
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)
        
        # Create layout
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw the graph
        plt.figure(figsize=(15, 10))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                              node_size=2000, alpha=0.7)
        
        # Draw edges
        edges = G.edges()
        weights = [G[u][v]['weight'] for u,v in edges]
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                              width=weights, arrowsize=20)
        
        # Add labels
        nx.draw_networkx_labels(G, pos, font_size=8)
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        
        plt.title("Category Learning Path Graph")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

def main():
    # Create graph instance
    category_graph = CategoryGraph()
    
    # Create visualizer instance
    visualizer = CategoryVisualizer(category_graph.graph)
    
    # Plot the graph
    visualizer.plot_graph()

if __name__ == "__main__":
    main() 