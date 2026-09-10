import torch
from torch import Tensor
from torch_scatter import scatter_mean
from pytorch3d.ops import knn_points
from ..utils.global_config import conf,EPS

class CVTLoss(torch.nn.Module):
    def __init__(self,stage:int):
        super(CVTLoss,self).__init__()
        self.stage=stage
        
    def get_site_wise_na_mean(self, pc: Tensor,normal: Tensor,sites: Tensor,num_seeds:Tensor) -> Tensor:
        '''
        pc,normal: [B,N,3]
        sites: [B,S,3]
        '''
        site_id = knn_points(pc,sites,lengths2=num_seeds, K=1, return_nn=False).idx.squeeze(-1)  # [B,N] 
        site_id_exp= site_id.unsqueeze(-1).expand(-1, -1, 3)  # [B,N,3]
        site_of_pc = torch.gather(sites, dim=1, index=site_id_exp)  # [B,N,3] site of each point

        na_dis = pc - site_of_pc  # [B,N,3]
        na_dis = torch.sum(normal * na_dis, dim=-1)**2  # [B,N]

        B,N,S = pc.shape[0], pc.shape[1], sites.shape[1]
        global_id = torch.arange(B, device=pc.device).unsqueeze(1).expand(-1, N)  # [B,N] (0,1,...,B-1)
        global_id = global_id * S + site_id
        global_id = global_id.reshape(-1)  # [B*N]

        na_dis = na_dis.reshape(-1)  # [B*N]
        na_mean = scatter_mean(na_dis, global_id, dim=0, dim_size=B*S)  # [B*S]
        na_mean = na_mean.view(B,S)  # [B,S]
        return na_mean
    
    def rel_na_loss(self, na1: Tensor, na2: Tensor, num_seeds:Tensor) -> Tensor:
        '''
        na1, na2: [B,S]
        '''
        def func(x):
            x = torch.where(x > 0, x, x)
            x = torch.clamp(x, -1, 1)
            return x

        d_na = na2 - na1  # [B,S]
        d_na = d_na / (na1 + 1e-8)
        d_na = func(d_na)
        ids = torch.arange(na1.shape[1],device=na1.device)  # [0 1 2...S-1]
        mask = ids < num_seeds.unsqueeze(1)  # [B,S] 'True' for valid sites
        d_na = d_na*mask.float()  # [B,S] set invalid distance to 0
        loss = torch.sum(d_na,dim=1) / torch.sum(mask, dim=1).float()  # [B]
        return loss
 
    def na_loss(self, pc: Tensor,normal: Tensor,sites: Tensor,num_seeds:Tensor) -> Tensor:
        '''
        pc,normal: [B,N,3]
        '''
        site_id = knn_points(pc,sites,lengths2=num_seeds, K=1, return_nn=False).idx.squeeze(-1) 
        site_id= site_id.unsqueeze(-1).expand(-1, -1, 3)  # [B,N,3]
        site_of_pc = torch.gather(sites, dim=1, index=site_id)  # [B,N,3] site of each point

        na_dis = pc - site_of_pc  # [B,N,3]
        na_dis = torch.sum(normal * na_dis, dim=-1)  # [B,N]
        loss=na_dis**2

        loss_mean = torch.mean(loss,dim=-1)  # [B]
        loss_max = torch.max(loss,dim=-1).values  # [B]
        return loss_mean,loss_max
    
    def cvt_loss(self,pc: Tensor,sites: Tensor,num_seeds:Tensor) -> Tensor:
        dis = knn_points(pc,sites,lengths2=num_seeds, K=1, return_nn=False).dists.squeeze(-1)  # dis is squared
        cvt_loss = torch.mean(dis,dim=1) 
        return cvt_loss

    def reg_loss(self,pc: Tensor,sites: Tensor,num_seeds:Tensor) -> Tensor:
        dis = knn_points(sites,pc,lengths1=num_seeds, K=1, return_nn=False).dists.squeeze(-1)  # dis is squared
        ids = torch.arange(dis.shape[1],device=dis.device)  # [0 1 2...S-1]
        mask = ids < num_seeds.unsqueeze(1)  # [B,S] 'True' for valid sites
        dis = dis*mask.float()  # [B,S] set invalid distance to 0
        reg_mean = torch.sum(dis,dim=1) / torch.sum(mask, dim=1).float()
        reg_max = torch.max(dis,dim=1).values
        return reg_mean,reg_max

    @torch.no_grad()
    def evaluate(self,pc: Tensor,normal: Tensor,sites:list,num_seeds:Tensor)->Tensor:
        batch_size= num_seeds.shape[0]
        pc = pc.reshape(batch_size,-1, 3)  # [B,N,3]
        normal = normal.reshape(batch_size,-1, 3)  # [B,N,3]

        cvt,reg_max,na_mean,site_wise_na=[0]*2,[0]*2,[0]*2,[0]*2
        for i in {0,-1}:
            cvt_split=self.cvt_loss(pc,sites[i],num_seeds)
            _,reg_split_max=self.reg_loss(pc,sites[i],num_seeds)
            cvt[i] = torch.mean(cvt_split)
            reg_max[i] = torch.mean(reg_split_max)

            if self.stage==2 or self.stage==3:
                na_split_mean,_=self.na_loss(pc,normal, sites[i],num_seeds)
                na_mean[i] = torch.mean(na_split_mean)
                site_wise_na[i]=self.get_site_wise_na_mean(pc,normal,sites[i],num_seeds)  # [B,S]

        rel_na=0
        if self.stage==3:
            rel_na_split = self.rel_na_loss(site_wise_na[0], site_wise_na[-1], num_seeds)
            rel_na = torch.mean(rel_na_split)
            
        return [cvt[-1],reg_max[-1],na_mean[-1],rel_na]

    def forward(self,pc: Tensor,normal: Tensor,sites:list,num_seeds:Tensor)->Tensor:
        '''
        pc: [B*N,3]
        sites: [B,num_samples (S),3]
        '''
        batch_size = num_seeds.shape[0]
        pc = pc.reshape(batch_size,-1, 3)  # [B,N,3]
        normal = normal.reshape(batch_size,-1, 3)  # [B,N,3]

        # loss: [CVT, Reg, NA, Rel]
        loss_w = torch.as_tensor(conf.train.loss.weight[self.stage], dtype=pc.dtype, device=pc.device)  # [num_losses]
        loss_enabled = [loss_w[i].abs() > EPS for i in range(len(loss_w))]  # [num_losses]

        loss_cvt,loss_reg_max,loss_na_mean,loss_rel_na,n,s = 0,0,0,0,len(sites),1
        if loss_enabled[3]:
            site_wise_na_mean=[0]*len(sites)
            for i in range(n):
                site_wise_na_mean[i]=self.get_site_wise_na_mean(pc,normal,sites[i],num_seeds)  # [B,S]

        for i in range(s,n):
            loss_cvt_split=num_seeds * self.cvt_loss(pc,sites[i],num_seeds)
            _,loss_reg_split_max=self.reg_loss(pc,sites[i],num_seeds)
            loss_cvt += torch.mean(loss_cvt_split)
            loss_reg_max += torch.mean(loss_reg_split_max)
            if loss_enabled[2]:
                loss_na_split_mean,_ =self.na_loss(pc,normal,sites[i], num_seeds)
                loss_na_mean += torch.mean(loss_na_split_mean)

            if loss_enabled[3]:
                loss_rel_na_split=self.rel_na_loss(site_wise_na_mean[i-1],site_wise_na_mean[i],num_seeds)
                loss_rel_na += torch.mean(loss_rel_na_split)

      
        loss=[loss_cvt,loss_reg_max,loss_na_mean,loss_rel_na]
        loss = [torch.as_tensor(l, dtype=pc.dtype, device=pc.device) for l in loss]
        loss=torch.stack(loss)  # [num_losses]
        loss = loss * loss_w / (n-s)  
        loss_tot= torch.sum(loss)
        return [loss_tot,loss[0],loss[1],loss[2],loss[3]]
