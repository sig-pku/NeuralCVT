import os
import torch
import torch_geometric
from src.data.dataset import TestMeshDataset
from src.network.our_network import CVT_Net
from src.utils.global_config import conf
from src.utils.common import *
import subprocess

def export_seeds(output_path, seeds, normalized_mesh):
    mesh,(b_mid,b_len)=normalized_mesh
    if conf.test.project_seeds: 
        seeds, _, _ = mesh.nearest.on_surface(points=seeds.cpu().numpy())
        seeds=pc_normalization_reverse(seeds,(b_mid,b_len))
    else:
        b_mid = torch.as_tensor(b_mid, dtype=torch.float32, device=seeds.device)
        b_len = torch.as_tensor(b_len, dtype=torch.float32, device=seeds.device)
        seeds=pc_normalization_reverse(seeds,(b_mid,b_len)).cpu().numpy()
    trimesh.points.PointCloud(seeds).export(output_path, file_type='obj')

# Not thread-safe; can't specify RDT filepath in Geogram
def extract_mesh(mesh_file_path,output_seeds_path,output_remesh_path):
    subprocess.run([conf.test.geogram_path + "/bin/compute_RVD",mesh_file_path,output_seeds_path,output_remesh_path,"RDT=true","RVD=false","volumetric=false"], capture_output=True, text=True)
    subprocess.run([conf.test.geogram_path + "/bin/vorpalite","RDT.meshb","RDT.obj","pre=false","remesh=false","post=false"], capture_output=True, text=True)
    subprocess.run(["mv","RDT.obj",output_remesh_path], capture_output=True, text=True)

def test_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(conf.gpu_id)
    set_random_seed(conf.random_seed)

    num_iters=[conf.test.num_iters[s+1] for s in range(3)]

    output_dir = os.path.join(conf.test.output_dir,f"{conf.name}(num_seeds={conf.test.num_seeds},size_pc={conf.test.size_pc},num_iters=[{','.join(map(str, num_iters))}],proj={conf.test.project_seeds})")
    assert not os.path.exists(output_dir), f"Output dir {output_dir} already exists."

    subdirs = ["remesh", "optimized_seeds"]
    dirs = {name: os.path.join(output_dir, name) for name in subdirs}
    for path in dirs.values(): os.makedirs(path, exist_ok=True)

    input_mesh=get_files_in_dir(conf.test.test_mesh_dir)
    normalized_mesh=[]
    for _,_,path in input_mesh:
        mesh = read_tri_mesh(path)
        mesh.vertices,(b_mid, b_len)= pc_normalization(mesh.vertices)
        normalized_mesh.append((mesh,(b_mid, b_len)))

    test_data = TestMeshDataset(input_mesh,normalized_mesh)

    cf=conf.test
    test_data = torch_geometric.loader.DataLoader(test_data, batch_size=cf.batch_size, shuffle=False, pin_memory=False,num_workers=cf.num_workers)

    models=[0,0,0]
    for s in range(3):
        if num_iters[s] <= 0: continue
        models[s]=CVT_Net()
        models[s].to(device)
        if cf.checkpoint.epoch[s+1] == -1:
            pkl_file_name = "latest.pkl"
        else:
            pkl_file_name = f"{cf.checkpoint.epoch[s+1]}.pkl"
        pkl_path = os.path.join(cf.checkpoint.dir[s+1], pkl_file_name)
        if os.path.exists(pkl_path):
            data=torch.load(pkl_path, map_location=device,weights_only=True)
            models[s].load_state_dict(data['model'])
            models[s].eval()
            for param in models[s].parameters():
                param.requires_grad = False
            print("Load model from: "+pkl_path)
        else:
            print(f"Checkpoint {pkl_path} does not exist.")
            return

    for graph_batch in test_data:
        graph_batch = graph_batch.to(device)
        with torch.no_grad():
            seeds=graph_batch.init_seeds
            for s in range(3):
                if num_iters[s] <= 0: continue
                res,seeds=models[s](graph_batch.pos,graph_batch.normal, graph_batch.edge_index, graph_batch.batch,seeds,graph_batch.num_seeds,num_iters[s],conf.test.num_seeds)
            
            seeds=res[-1]
            for b in range(graph_batch.num_graphs):
                output_seeds_path = os.path.join(dirs["optimized_seeds"], graph_batch.mesh_name[b] + ".obj")
                export_seeds(output_seeds_path, seeds[b], normalized_mesh[graph_batch.id[b]])
                output_remesh_path = os.path.join(dirs["remesh"], graph_batch.mesh_name[b] + ".obj")
                extract_mesh(graph_batch.mesh_file_path[b],output_seeds_path,output_remesh_path)
