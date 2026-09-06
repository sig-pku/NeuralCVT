import torch
from torch import Tensor
import torch_geometric
import torch.nn.functional as F
from .graph_unet import GraphUNet
from .positional_encoder import PositionalEncoder
from ..utils.global_config import conf,EPS

class MLP(torch.nn.Module):
    def __init__(self, in_channels: int,hidden_channels: int, out_channels: int,num_layers: int):
        super(MLP, self).__init__()
        assert num_layers >= 2, "MLP must have at least 2 layers"
        self.num_layers = num_layers
        self.fc= torch.nn.ModuleList()
        self.fc.append(torch.nn.Linear(in_channels, hidden_channels, bias=True))
        for i in range(1,num_layers-1):
            self.fc.append(torch.nn.Linear(hidden_channels+(i%3==0)*in_channels, hidden_channels, bias=True))
        self.fc.append(torch.nn.Linear(hidden_channels+((num_layers-1)%3==0)*in_channels, out_channels, bias=True))
        self.relu = torch.nn.ReLU()

    def forward(self, input_feature: Tensor) -> Tensor:
        feature=input_feature
        for i in range(0,self.num_layers):
            if i > 0 and i%3==0:
                feature = torch.cat([feature, input_feature], dim=-1)
            feature=self.fc[i](feature)
            if i < self.num_layers-1:
                feature = self.relu(feature)
        return feature
    

class InputFeature(torch.nn.Module):
    def __init__(self):
        super(InputFeature, self).__init__()
        cf=conf.network.input_feature
        self.in_channels=3+3+(cf.pos.num_pe_freqs+cf.normal.num_pe_freqs) * 6
        self.pe_pos = PositionalEncoder(num_freqs=conf.network.input_feature.pos.num_pe_freqs)  # [B*N,36]
        self.pe_normal = PositionalEncoder(num_freqs=conf.network.input_feature.normal.num_pe_freqs)

    def forward(self, pos: Tensor, normal: Tensor) -> Tensor:
        pe_pos = self.pe_pos(pos)
        pe_normal = self.pe_normal(normal)
        feature=torch.cat([pos, normal,pe_pos,pe_normal], dim=1)  # better to include the original input
        return feature

class CVT_Net(torch.nn.Module):
    def __init__(self):
        super(CVT_Net, self).__init__()
        self.input_feature = InputFeature()

        cf=conf.network.graph_unet
        self.graph_unet = GraphUNet(self.input_feature.in_channels, cf.out_channels, cf.depth,cf.num_res_blocks_per_layer)

        cf=conf.network.sub_graph_unet
        self.sub_graph_unet = GraphUNet(cf.in_channels, cf.out_channels, cf.depth,cf.num_res_blocks_per_layer)

        self.knn_graph = torch_geometric.transforms.KNNGraph(k=conf.network.sub_graph_unet.knn)
        in_channels = conf.network.graph_unet.out_channels \
                + 3*conf.network.gru.use_proj_feature \
                + 3*conf.network.gru.concat_pos
        self.gru_cell=torch.nn.GRUCell(input_size=in_channels,hidden_size=conf.network.gru.hidden_channels)
        cf=conf.network.mlp
        self.mlp= MLP(in_channels=conf.network.gru.hidden_channels,hidden_channels=cf.hidden_channels,out_channels=3,num_layers=cf.num_layers)

    def GRU(self, pc_feature: Tensor, pc: Tensor, pc_normal:Tensor,pc_batch: Tensor,init_seeds:Tensor,num_seeds:Tensor,num_iters:int,padding_size: int) -> Tensor:
        """
        pc_feature: [B*N,F]
        pc, pc_normal: [B*N,3]
        pc_batch: [B*N]
        init_seeds: [B*S,3]
        """
        batch_size = pc_batch.max().item() + 1
        sites=[0]*(num_iters+1)
        sites[0]=init_seeds
        batch = torch.repeat_interleave(torch.arange(batch_size ,device=pc.device), repeats=num_seeds)
        h = torch.zeros(sites[0].shape[0], conf.network.gru.hidden_channels, dtype=pc.dtype,device=pc.device)  # [B*S,F] h0=0

        for t in range(1,num_iters+1):
            pc_sampled = torch_geometric.data.Data(pos=sites[t-1],batch=batch)
            sub_graph = self.knn_graph(pc_sampled)
            feature_interp = torch_geometric.nn.knn_interpolate(torch.cat([pc_feature,pc_normal,pc],dim=-1), pc, sites[t-1],batch_x=pc_batch, batch_y=batch, k=3)
            graph_feature,normal,proj=feature_interp[:,:-6],feature_interp[:,-6:-3],feature_interp[:,-3:]
            normal = F.normalize(normal, p=2, dim=1, eps=EPS)
            x = self.sub_graph_unet(graph_feature,sites[t-1], normal, sub_graph.edge_index, batch)
            if conf.network.gru.concat_pos:
                x = torch.cat([x, sites[t-1]], dim=-1)
            if conf.network.gru.use_proj_feature:
                x= torch.cat([x, proj-sites[t-1]], dim=-1)
            h = self.gru_cell(x,h)
            delta_pos = self.mlp(h)  # [B*S,3]
            sites[t]=sites[t-1]+delta_pos  # [B*S,3]

        res=sites[-1].clone()
        for i in range(len(sites)):
            padding = torch.full((batch_size, padding_size, 3), 10, dtype=pc.dtype,device=pc.device)  # [B,S,3] use 10 as padding value, 'torch.finfo(pc.dtype).max' will crash in computation
            for b in range(batch_size):
                padding[b, :num_seeds[b], :] = sites[i][batch == b]
            sites[i]= padding
        return sites,res
    
    def forward(self, pc: Tensor, normal:Tensor, edge_index: Tensor,batch: Tensor,init_seeds:Tensor,num_seeds:Tensor,num_iters:int,padding_size: int) -> Tensor:
        """
        pos should be in [-1,1]
        pc: [B*N,3]
        init_seeds: [B*num_samples (S),3]
        num_seeds: [B]
        """
        feature= self.input_feature(pc,normal)
        feature = self.graph_unet(feature,pc, normal, edge_index, batch)  # [B*N,256]
        sites = self.GRU(feature, pc, normal, batch, init_seeds,num_seeds,num_iters,padding_size)
        return sites