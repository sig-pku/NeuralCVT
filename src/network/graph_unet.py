import torch
from torch import Tensor
import torch_geometric
from .pooling import GridBasedPooling, Unpooling
from .res_block import GraphResBlock

class GraphUNet(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int ,depth: int,num_res_blocks_per_layer:int):
        super(GraphUNet, self).__init__()
        self.depth=depth
        self.in_conv=GraphResBlock(in_channels,out_channels,out_channels)
        bottle_neck=4
        self.num_res_blocks=num_res_blocks_per_layer
        encoder_channel = [out_channels]*depth
        self.res_block_down= torch.nn.ModuleList(torch.nn.ModuleList() for _ in range(depth-1))
        for d in range(depth-1):
            for _ in range(num_res_blocks_per_layer):
                self.res_block_down[d].append(GraphResBlock(encoder_channel[d],encoder_channel[d]//bottle_neck,encoder_channel[d]))

        decoder_channel = [out_channels]*depth
        self.res_block_up= torch.nn.ModuleList(torch.nn.ModuleList() for _ in range(depth-1)) 
        for d in range(depth-1):
            self.res_block_up[d].append(GraphResBlock(decoder_channel[d]*2,decoder_channel[d]//bottle_neck,decoder_channel[d]))
            for _ in range(num_res_blocks_per_layer-1):  # first layer is different
                self.res_block_up[d].append(GraphResBlock(decoder_channel[d],decoder_channel[d]//bottle_neck,decoder_channel[d]))

        self.pooling = GridBasedPooling()
        self.unpooling = Unpooling()
        self.out_conv=torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels, bias=False),
            torch.nn.GroupNorm(num_groups=4, num_channels=out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels, bias=True),
        )

    def get_embedding(self, feature:Tensor, pos: Tensor, normal:Tensor, edge_index: Tensor,batch: Tensor) -> Tensor:
        """
        graph at a depth of d (d>0): aggregate vertices in voxel with side length = 2/2**(max_depth-d) 
        """
        feature = self.in_conv(f=feature, v=pos, edge_index=edge_index)
        
        graph=[]  # [depth] each layer of graph
        graph.append(torch_geometric.data.Data(
            x=feature,
            pos=pos, 
            normal=normal, 
            edge_index=edge_index,
            batch=batch))  # graph[0]

        for d in range(1,self.depth):
            voxel_len_pos = 2 / (2 ** (self.depth - d))  # 2/2**(max_depth-d)
            voxel_len_n=voxel_len_pos*3
            voxel_len=[voxel_len_pos]*3+[voxel_len_n]*3
            pooled_graph = self.pooling(graph[d-1],voxel_len)
            graph.append(pooled_graph)  # graph[d]
            for i in range(self.num_res_blocks):
                graph[d].x= self.res_block_down[d-1][i](f=graph[d].x, v=graph[d].pos, edge_index=graph[d].edge_index)
            
        feature=graph[self.depth-1].x
        for d in range(self.depth-1,0,-1):
            feature = self.unpooling(feature, graph[d].cluster)  # [N, 256]
            feature = torch.cat([graph[d-1].x,feature], dim=1)  # [N, 512]
            for i in range(self.num_res_blocks):
                feature = self.res_block_up[d-1][i](f=feature, v=graph[d-1].pos, edge_index=graph[d-1].edge_index)

        return self.out_conv(feature)  # [N, 256]


    def forward(self, input_feature:Tensor, pos: Tensor, normal:Tensor, edge_index: Tensor,batch: Tensor) -> Tensor:
        """
        pos should be in [-1,1]
        """
        embedding = self.get_embedding(input_feature,pos, normal, edge_index, batch)  # [N, 256]
        return embedding
