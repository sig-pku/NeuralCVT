from tqdm import tqdm
import os
import torch
import torch_geometric
from torch.utils.tensorboard import SummaryWriter
from .data.dataset import MeshGraphDataset
from .network.our_network import CVT_Net
from .network.loss_function import CVTLoss
from .network.lr_scheduler import LR_Scheduler
from .utils.global_config import conf
from .utils.common import *

writers = {}
def write_log(dir_name, label, y, x):
    """
    dir_name: subdirectory name
    label: figure name
    """
    if dir_name not in writers:
        output_dir=os.path.join(conf.logging.log_dir,dir_name)
        writers[dir_name] = SummaryWriter(log_dir=output_dir, flush_secs=20)
    writers[dir_name].add_scalar(label, y, x)

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(conf.gpu_id)
    set_random_seed(conf.random_seed)

    cf=conf.train
    train_data = MeshGraphDataset(conf.preprocessing.training_file_dir,cf.filelist_path,cf.load_full_dataset,cf.size_dataset,cf.num_seeds_min,cf.num_seeds_max)
    train_data = torch_geometric.loader.DataLoader(train_data, batch_size=cf.batch_size, shuffle=True, pin_memory=cf.pin_memory,num_workers=cf.num_workers)

    val_data=[]
    for cf in conf.val:
        data= MeshGraphDataset(conf.preprocessing.training_file_dir,cf.filelist_path,cf.load_full_dataset,cf.size_dataset,cf.num_seeds_min,cf.num_seeds_max)
        data = torch_geometric.loader.DataLoader(data, batch_size=cf.batch_size, shuffle=False, pin_memory=cf.pin_memory,num_workers=cf.num_workers)
        val_data.append(data)

    model=CVT_Net()
    start_epoch=0
    cnt_iter=1

    model.to(device)
    cf=conf.train
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=cf.optimizer.lr,weight_decay=cf.optimizer.weight_decay)  # AdamW optimizer
    scheduler = LR_Scheduler(optimizer, num_epochs=cf.num_epochs[conf.stage], decay_type=cf.scheduler.decay_type, warmup_ratio=cf.scheduler.warmup_ratio)
    loss_function = CVTLoss(conf.stage)
    
    strategy=conf.train.strategy[conf.stage]
    os.makedirs(conf.logging.model_info_dir, exist_ok=True)
    latest_pkl_path =os.path.join(conf.logging.model_info_dir,"latest.pkl")
    if os.path.exists(latest_pkl_path):
        data=torch.load(latest_pkl_path, map_location=device,weights_only=True)
        model.load_state_dict(data['model'])
        optimizer.load_state_dict(data['optimizer'])
        scheduler.load_state_dict(data['scheduler'])
        start_epoch=data['epoch']
        cnt_iter=data['cnt_iter']
        print("Load model from: "+latest_pkl_path)
        print("Start epoch: "+str(start_epoch))
    elif strategy.train_from_scratch is False:
        pretrained_pkl_path=strategy.pretrained_model
        data=torch.load(pretrained_pkl_path, map_location=device,weights_only=True)
        model.load_state_dict(data['model'])
        print("Load model from: "+pretrained_pkl_path)

    pre_inference_iters = [0,0]
    if "pre_inference_s1" in strategy:
        model_s1=CVT_Net()
        model_s1.to(device)
        data=torch.load(strategy.pre_inference_s1.model, map_location=device,weights_only=True)
        model_s1.load_state_dict(data['model'])
        print("Load pre-inference model (stage 1) from: "+strategy.pre_inference_s1.model)
        model_s1.eval()
        pre_inference_iters[0]=strategy.pre_inference_s1.num_iters
        for param in model_s1.parameters():
            param.requires_grad = False

    if "pre_inference_s2" in strategy:
        model_s2=CVT_Net()
        model_s2.to(device)
        data=torch.load(strategy.pre_inference_s2.model, map_location=device,weights_only=True)
        model_s2.load_state_dict(data['model'])
        print("Load pre-inference model (stage 2) from: "+strategy.pre_inference_s2.model)
        model_s2.eval()
        pre_inference_iters[1]=strategy.pre_inference_s2.num_iters
        for param in model_s2.parameters():
            param.requires_grad = False

    data_labels_train= (["loss/[a] Total", "loss/[b] CVT",
                         "loss/[c] Reg", "loss/[d] NA", "loss/[e] Rel_NA"])
    
    data_labels_val= (["evaluation/[a] CVT","evaluation/[b] Reg",
                       "evaluation/[c] NA","evaluation/[d] Rel_NA"])

    for epoch in range(start_epoch,conf.train.num_epochs[conf.stage]):
        # training process
        model.train()  # enable batch normalization and dropout
        loss_function.train()  # for self.training
        log_data=[0]*len(data_labels_train)
        total_samples=0
        optimizer.zero_grad()  # initialize the gradient before accumulation
        for i,graph_batch in enumerate(tqdm(train_data, position=0, leave=False, ncols=60,desc=f'Training (epoch {epoch})')):
            graph_batch = graph_batch.to(device)
            n=random.randint(conf.train.num_iters_min,conf.train.num_iters_max)
            seeds=graph_batch.init_seeds
            with torch.no_grad():
                if pre_inference_iters[0] > 0:
                    _,seeds=model_s1(graph_batch.pos,graph_batch.normal, graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,pre_inference_iters[0],conf.train.num_seeds_max)
                if pre_inference_iters[1] > 0:
                    _,seeds=model_s2(graph_batch.pos,graph_batch.normal, graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,pre_inference_iters[1],conf.train.num_seeds_max)

                aug_start_epoch=conf.train.data_augmentation.start_epoch[conf.stage]
                it_max=conf.train.data_augmentation.forward_iters_max[conf.stage]
                if it_max > 0 and epoch >= aug_start_epoch:
                    cur_max=int((epoch-aug_start_epoch)*it_max/(conf.train.num_epochs[conf.stage]-aug_start_epoch))
                    forward_iters=random.randint(0,cur_max)
                    if forward_iters > 0:
                        _,seeds=model(graph_batch.pos,graph_batch.normal, graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,forward_iters,conf.train.num_seeds_max)

            sites,_=model(graph_batch.pos,graph_batch.normal, graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,n,conf.train.num_seeds_max)  # [B,S,3]
            loss= loss_function(graph_batch.pos,graph_batch.normal,sites,graph_batch.num_seeds)

            total_samples+=graph_batch.num_graphs
            for j in range(len(loss)):
                loss_j=loss[j]
                if isinstance(loss_j, torch.Tensor):
                    loss_j= loss_j.item()
                log_data[j]+=loss_j*graph_batch.num_graphs

            loss[0] /= conf.train.grad_accum_steps  # average the loss for gradient accumulation
            loss[0].backward()  # calculate the gradient

            if (i+1) % conf.train.grad_accum_steps == 0 or (i+1) == len(train_data):
                cnt_iter+=1
                if conf.train.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=conf.train.grad_clip_norm)  # clip the gradient
                optimizer.step()  # update the parameters
                optimizer.zero_grad()

        for i in range(len(data_labels_train)):
            write_log("train",data_labels_train[i], log_data[i]/len(train_data.dataset), epoch+1)

        scheduler.step()

        res_data={"model": model.state_dict(),'optimizer': optimizer.state_dict(),'scheduler': scheduler.state_dict(),"epoch": epoch+1,"cnt_iter": cnt_iter}
        torch.save(res_data, latest_pkl_path)
        if (epoch+1) % conf.logging.save_model_every_epoches == 0 and epoch > 0:
            pkl_path =os.path.join(conf.logging.model_info_dir,f"{epoch+1}.pkl")
            torch.save(res_data, pkl_path)
   
        # validation process
        model.eval()
        loss_function.eval()  # for self.training
        with torch.no_grad():
            for v in range(len(val_data)):
                log_data=[0]*len(data_labels_val)
                for i,graph_batch in enumerate(tqdm(val_data[v], position=0, leave=False, ncols=60,desc=f'Validating {v} (epoch {epoch})')):
                    graph_batch = graph_batch.to(device)
                    seeds=graph_batch.init_seeds
                    if conf.val[v].pre_inference_iters[conf.stage][0] > 0:
                        _,seeds=model_s1(graph_batch.pos,graph_batch.normal,graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,conf.val[v].pre_inference_iters[conf.stage][0],conf.val[v].num_seeds_max)
                    if conf.val[v].pre_inference_iters[conf.stage][1] > 0:
                        _,seeds=model_s2(graph_batch.pos,graph_batch.normal,graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,conf.val[v].pre_inference_iters[conf.stage][1],conf.val[v].num_seeds_max)
                    
                    sites,_=model(graph_batch.pos,graph_batch.normal,graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,conf.val[v].num_iters[conf.stage],conf.val[v].num_seeds_max)  # [B,num_samples,3]
                    evaluation = loss_function.evaluate(graph_batch.pos,graph_batch.normal,sites,graph_batch.num_seeds)

                    for j in range(len(evaluation)):
                        if isinstance(evaluation[j], torch.Tensor):
                            evaluation[j]= evaluation[j].item()
                        log_data[j]+=evaluation[j]*graph_batch.num_graphs

                for i in range(len(data_labels_val)):
                    write_log(f"val{v}",data_labels_val[i], log_data[i]/len(val_data[v].dataset), epoch+1)
                    
    for w in writers.values(): w.close()