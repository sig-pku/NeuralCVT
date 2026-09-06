import os
import random
import trimesh
import torch
import numpy as np
import multiprocessing as mp
import torch.nn.functional as F
from src.utils.common import *
from tqdm import tqdm

def generate_training_data(mesh: trimesh.Trimesh, size_pc,even_sample=True):
    if even_sample:
        points, f_id = trimesh.sample.sample_surface_even(mesh, count=size_pc)
        if points.shape[0] < size_pc:
            need = size_pc - points.shape[0]
            p2, f2 = trimesh.sample.sample_surface(mesh, count=need)
            points = np.concatenate([points, p2], axis=0)
            f_id = np.concatenate([f_id, f2], axis=0)
        elif points.shape[0] > size_pc:
            points = points[:size_pc]
            f_id = f_id[:size_pc]
    else:
        points, f_id = trimesh.sample.sample_surface(mesh, count=size_pc)

    normals = mesh.face_normals[f_id]
    points = torch.from_numpy(points.astype(np.float32))  # [V,3]
    normals = torch.from_numpy(normals.astype(np.float32))  # [V,3]
    normals = F.normalize(normals, p=2, dim=1, eps=1e-12)

    data = {"points": points,"normals": normals}
    return data

def generate_training_data_file(mesh_file,output_file,size_pc):
    try:
        mesh = read_tri_mesh(mesh_file)
    except Exception as e:
        print("Fail to read file: "+mesh_file)
        print("Error: ", e)
        return False

    if len(mesh.vertices)==0 or len(mesh.faces)==0:
        print("Empty file: "+mesh_file)
        return False
    
    mesh.vertices,_= pc_normalization(mesh.vertices)

    try:
        data = generate_training_data(mesh, size_pc)
    except Exception as e:
        print("Fail to sample point cloud: "+mesh_file)
        print("Error: ", e)
        return False
    
    if data["points"].shape[0]!=size_pc:
        print("Unmatched point cloud size: "+mesh_file)
        return False
    
    torch.save(data,output_file)
    return True
    
def generate_training_data_dir(files_dir,output_dir,size_pc,parallel=True):
    files=get_files_in_dir(files_dir)
    assert not os.path.exists(output_dir), f"Output dir {output_dir} already exists."
    os.makedirs(output_dir, exist_ok=True)
    mesh_file_ext = ['.obj', '.off', '.stl', '.ply']
    params = []
    for i,(file_name,file_ext,file_path) in enumerate(files):
        if file_ext in mesh_file_ext:
            output_file=os.path.join(output_dir,file_name+".pt")
            if not os.path.exists(output_file):
                params.append((file_path,output_file,size_pc))
    
    if parallel:
        with mp.Pool(mp.cpu_count()) as pool:  
            pool.starmap(generate_training_data_file, params)
    else:
        for i,(file_path,output_file,size_pc) in enumerate(tqdm(params, position=0, leave=False, ncols=60)):
            generate_training_data_file(file_path,output_file,size_pc)

def split_dataset(data_dir,filelist_dir,ratio=0.8):
    files=get_files_in_dir(data_dir)
    assert not os.path.exists(filelist_dir), f"Output dir {filelist_dir} already exists."
    os.makedirs(filelist_dir, exist_ok=True)
    random.shuffle(files)
    num_train = int(len(files) * ratio)
    files_train = files[:num_train]
    files_test  = files[num_train:]
    
    with open(os.path.join(filelist_dir,'filelist_train.txt'), 'w') as f:
        for file in files_train:
            f.write(file[0]+file[1] + '\n')

    with open(os.path.join(filelist_dir,'filelist_val.txt'), 'w') as f:
        for file in files_test:
            f.write(file[0]+file[1] + '\n')
