import torch
from torch import Tensor
import torch_geometric

class GridBasedPooling(torch.nn.Module):
    def __init__(self):
        super(GridBasedPooling, self).__init__()

    def forward(self, graph:torch_geometric.data.Data, voxel_len: Tensor) -> torch_geometric.data.Data:
        pos_and_normal=torch.cat([graph.pos, graph.normal], dim=1)
        # PyG separates batches by adding a batch index to the data
        # cluster is the voxel id for each point; ids can exceed N
        cluster = torch_geometric.nn.voxel_grid(
            pos=pos_and_normal, 
            size=voxel_len, 
            batch=graph.batch,
            start=[-1, -1, -1.], 
            end=[1, 1, 1.])  # [N]
        _, cluster = torch.unique(cluster, return_inverse=True)  # [N]
        # pooling also averages pos/normal, so include them in x for pooling
        graph4pooling = torch_geometric.data.Data(
            x=torch.cat([graph.x, pos_and_normal], dim=1),
            edge_index=graph.edge_index, 
            batch=graph.batch)
        # pooled data uses merged x/edge_index/batch; pos is averaged
        pooled_graph = torch_geometric.nn.avg_pool(cluster, graph4pooling)
        return torch_geometric.data.Data(
            x=pooled_graph.x[:,:-6],
            pos=pooled_graph.x[:,-6:-3], 
            normal=pooled_graph.x[:,-3:], 
            cluster=cluster,
            edge_index=pooled_graph.edge_index,
            batch=pooled_graph.batch)

class Unpooling(torch.nn.Module):
    def __init__(self):
        super(Unpooling, self).__init__()

    def forward(self, feature: Tensor,cluster:Tensor) -> Tensor:
        """
        feature: [N (after pooling), D]
        cluster: [N (before pooling)]
        """
        cluster = cluster.unsqueeze(-1).repeat(1, feature.shape[1])  # [N (before pooling), D]
        feature = torch.gather(input=feature, dim=0, index=cluster)  # [N (before pooling), D]
        return feature
