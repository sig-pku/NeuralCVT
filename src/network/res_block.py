import torch
from torch import Tensor
import torch_geometric


class Conv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super(Conv, self).__init__(aggr='max')
        self.l1 = torch.nn.Linear(in_channels, out_channels)
        self.l2 = torch.nn.Linear(in_channels+3+1, out_channels)

    def forward(self, f: Tensor, v: Tensor, edge_index: Tensor)-> Tensor:   
        """
        f: feature [N, in_channels]
        v: position [N, 3]
        edge_index: [2,E]
        """
        return self.propagate(edge_index, f=f,v=v)  # [N, out_channels]

    def message(self, f_j: Tensor,v_i: Tensor,v_j: Tensor)-> Tensor: 
        """
        W1 x [f_j || v_j-v_i || dist_ij]
        message can take tensors from propagate() and map to _i, _j
        f_i, f_j: [E, in_channels]
        v_i, v_j: [E, 3]
        """
        vec=v_j - v_i  # [E,3]
        vec_norm = torch.norm(vec, dim=1, keepdim=True)  # [E,1]
        x = torch.cat([f_j, vec, vec_norm], dim=-1)  # [E,in_channels+3+1]
        return self.l2(x)  # [E, out_channels]

    def update(self, aggr_out, f):
        """
        W0 x f_i + aggr_out
        update's first arg is the aggregation result; it can take propagate() args
        aggr_out: [N, out_channels]
        f: [E, 6] same as forward input
        """
        return self.l1(f) + aggr_out  # [N, out_channels]


class GraphResBlock(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super(GraphResBlock,self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = Conv(in_channels, hidden_channels)
        self.conv2 = Conv(hidden_channels, out_channels)
        self.bn1 = torch.nn.GroupNorm(num_groups=4, num_channels=hidden_channels)
        self.bn2 = torch.nn.GroupNorm(num_groups=4, num_channels=out_channels)
        self.relu = torch.nn.ReLU()
        if self.in_channels != self.out_channels:
            self.conv3 = torch.nn.Linear(in_channels, out_channels)
            self.bn3 = torch.nn.GroupNorm(num_groups=4, num_channels=out_channels)

    def forward(self, f, v, edge_index):
        input_f=f  # [N, input_channels]
        f = self.conv1(f, v, edge_index)  # [N, hidden_channels]
        f = self.bn1(f)
        f = self.relu(f)
        f = self.conv2(f, v, edge_index)  # [N, out_channels]
        f = self.bn2(f)
        if self.in_channels != self.out_channels:
            input_f = self.conv3(input_f)
            input_f = self.bn3(input_f)
        f = self.relu(f+input_f)
        return f  # [N, out_channels]