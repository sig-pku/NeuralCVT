import os
import random
import trimesh
import torch
import numpy as np

def get_files_in_dir(dir_path):
    files = []
    for root, dir_names, file_names in os.walk(dir_path):
        dir_names.sort()
        for file_name in sorted(file_names):
            file_path = os.path.join(root, file_name)
            name,ext=os.path.splitext(file_name)
            files.append((name, ext, file_path))
    return files

def get_filepaths_from_filelist(filelist):
    filepaths = []
    with open(filelist, 'r') as f:
        lines = f.readlines()
    for line in lines:
        filepaths.append(line.strip('\n'))
    return filepaths

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def read_tri_mesh(obj_file):
    mesh = trimesh.load_mesh(obj_file,process=False)
    if isinstance(mesh, trimesh.Scene):  # merge all geometry into one mesh
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    mesh.visual.material = None
    mesh.merge_vertices()
    return mesh

def pc_normalization(points:np.ndarray):
    b_min = np.min(points, axis=0)
    b_max = np.max(points, axis=0)
    b_mid = (b_max + b_min) / 2
    b_len = b_max - b_min
    b_len = np.max(b_len)/2
    points = (points - b_mid) / b_len
    return points,(b_mid, b_len)

def pc_normalization_reverse(points,para):
    b_mid, b_len = para
    points = points * b_len + b_mid
    return points

