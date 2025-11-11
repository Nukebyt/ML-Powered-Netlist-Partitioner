import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from itertools import combinations

# --- GNN MODEL DEFINITION ---
class PartitionGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        """
        Defines the GNN architecture.
        num_node_features: The number of features for each node (e.g., 2 for degree and avg_net_size)
        """
        super().__init__()
        # Using the improved 2-feature, wider/deeper model
        self.conv1 = GCNConv(num_node_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.output_layer = Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.output_layer(x)
        return x

# --- DATA CONVERTER ---
def convert_to_graph_data(netlist, cells, partition=None):
    """
    Converts a netlist into a PyTorch Geometric Data object.
    'partition' is optional and only used for training.
    """
    cell_map = {name: i for i, name in enumerate(cells)}
    num_cells = len(cells)

    # Feature Engineering (Degree, Avg Net Size)
    features = []
    for cell_name in cells:
        cell_nets = [net for net in netlist if cell_name in net]
        degree = len(cell_nets)
        if not cell_nets:
            avg_net_size = 0
        else:
            avg_net_size = sum(len(net) for net in cell_nets) / len(cell_nets)
        features.append([degree, avg_net_size])
    
    x = torch.tensor(features, dtype=torch.float)

    # Edge Index (Clique Expansion)
    edge_list = []
    for net in netlist:
        cell_indices = [cell_map.get(c) for c in net if c in cell_map]
        for edge in combinations(cell_indices, 2):
            edge_list.append(list(edge))
            edge_list.append(list(reversed(edge))) # Add reverse edges for undirected graph
    
    # Handle case where there are no edges
    if not edge_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Labels (y) - only if a partition is provided (for training)
    y = None
    if partition:
        y = torch.zeros(num_cells, 1, dtype=torch.float)
        for cell_name in partition['B']:
            if cell_name in cell_map:
                y[cell_map[cell_name]] = 1.0
    
    return Data(x=x, edge_index=edge_index, y=y)

# --- PREDICTION FUNCTION ---
def predict_initial_partition(netlist, cells, model, device):
    """
    Uses the trained GNN to predict a balanced initial partition.
    """
    model.eval()
    
    # Convert the new netlist to the graph data format
    graph_data = convert_to_graph_data(netlist, cells, partition=None).to(device)

    # Get predictions from the model
    with torch.no_grad():
        logits = model(graph_data)
    
    # Squeeze to remove extra dimension
    probs = torch.sigmoid(logits.squeeze())
    
    # Ensure a balanced partition
    num_in_partition_A = len(cells) // 2
    
    # Get the indices of the cells with the lowest probability (most likely to be in A)
    _, top_indices_A = torch.topk(probs, k=num_in_partition_A, largest=False)
    top_indices_A = top_indices_A.cpu().numpy()

    # Create the partition dictionary
    initial_partition = {'A': [], 'B': []}
    for i, cell_name in enumerate(cells):
        if i in top_indices_A:
            initial_partition['A'].append(cell_name)
        else:
            initial_partition['B'].append(cell_name)
    
    return initial_partition