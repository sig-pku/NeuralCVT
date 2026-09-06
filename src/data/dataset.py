import os
import torch
import random
import torch_geometric
from src.utils.global_config import conf
from src.data.preprocessing import generate_training_data
from src.utils.common import *

class MeshGraphDataset(torch_geometric.data.Dataset):  # similar to torch.utils.data.Dataset
    def __init__(self, training_file_dir,filelist,load_full_dataset,size_dataset,num_seeds_min,num_seeds_max):
        super(MeshGraphDataset, self).__init__()
        self.training_file_dir = training_file_dir
        self.num_seeds_min = num_seeds_min
        self.num_seeds_max = num_seeds_max
        self.file_name = get_filepaths_from_filelist(filelist)
        if not load_full_dataset:
            assert size_dataset <= len(self.file_name), "size_dataset is too large"
            self.file_name = self.file_name[:size_dataset]
        self.knn_graph = torch_geometric.transforms.KNNGraph(k=conf.network.graph_unet.knn)

    def get(self, id):
        return self.get_data(self.file_name[id],id)

    def len(self):
        return len(self.file_name)

    def generate_init_seeds(self, points, num_seeds):
        random_samples = torch.randperm(points.shape[0])[:num_seeds]
        init_seeds = points[random_samples]
        return init_seeds
    
    def get_data(self,file_name,id):
        file_path=os.path.join(self.training_file_dir,file_name)
        data=torch.load(file_path,map_location=torch.device('cpu'),weights_only=True)
        points = data['points']  # [V,3]
        normals = data['normals']  # [V,3]
        num_seeds = random.randint(self.num_seeds_min, self.num_seeds_max)
        init_seeds = self.generate_init_seeds(points,num_seeds)

        pc = torch_geometric.data.Data(pos=points)
        graph_info = self.knn_graph(pc)
        graph_info.normal = normals
        graph_info.init_seeds = init_seeds
        graph_info.num_seeds = num_seeds
        graph_info.id = id
        return graph_info

class TestMeshDataset(torch_geometric.data.Dataset):  # similar to torch.utils.data.Dataset
    def __init__(self, input_mesh,normalized_mesh):
        super(TestMeshDataset, self).__init__()
        self.input_mesh = input_mesh
        self.normalized_mesh = normalized_mesh
        self.knn_graph = torch_geometric.transforms.KNNGraph(k=conf.network.graph_unet.knn)

    def get(self, id):
        return self.get_data(id, self.input_mesh[id][0], self.input_mesh[id][2])

    def len(self):
        return len(self.input_mesh)

    def generate_init_seeds(self, points):
        random_samples = torch.randperm(points.shape[0])[:conf.test.num_seeds]
        init_seeds = points[random_samples]
        return init_seeds
    
    def get_data(self,id, mesh_name,mesh_file):
        mesh,_=self.normalized_mesh[id]
        data = generate_training_data(mesh,conf.test.size_pc)
        points = data['points']  # [V,3]
        init_seeds = self.generate_init_seeds(points)
        pc = torch_geometric.data.Data(pos=points)
        graph_info = self.knn_graph(pc)
        graph_info.id = id
        graph_info.normal = data['normals']  # [V,3]
        graph_info.init_seeds = init_seeds
        graph_info.mesh_name = mesh_name
        graph_info.mesh_file_path = mesh_file
        graph_info.num_seeds = conf.test.num_seeds
        return graph_info
    