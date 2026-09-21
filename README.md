<h1 align="center">
    Neural Centroidal Voronoi Tessellations
</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2609.08497"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
    <a href='https://martin-jc-xu.github.io/projects/NCVT/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
</p>

This repository is the official code release for the paper "Neural Centroidal Voronoi Tessellations". This paper is published in ACM Transactions on Graphics (SIGGRAPH ASIA 2026).

## Environment Setup

This project uses Conda for environment management. The setup has been tested on Linux servers without `sudo` access.

This repository uses [Git LFS](https://git-lfs.com/) to store model checkpoints. Install and initialize Git LFS before cloning:

```bash
conda create -n git-lfs-env git-lfs -y
conda activate git-lfs-env
git lfs install
```

Clone the repository and initialize all submodules:

```bash
git clone --recurse-submodules https://github.com/sig-pku/NeuralCVT.git
cd NeuralCVT
```

If CMake is not already available in your environment, create a dedicated Conda environment and build Geogram as follows:

```bash
conda create -n cmake-env cmake -y
conda activate cmake-env
conda install -y gcc_linux-64 gxx_linux-64

cd geogram && mkdir build && cd build
cmake ..  -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \
          -DCMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
          -DGEOGRAM_WITH_GRAPHICS=OFF \
          -DGEOGRAM_WITH_LUA=OFF
make -j$(nproc)

cd ../..
```

Create the Python environment:

```bash
conda create -n ncvt python=3.10 -y
conda activate ncvt
```

Install PyTorch:

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```

Install PyTorch Geometric:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
pip install torch_geometric==2.6.1
```

Install PyTorch3D:

```bash
pip install iopath
conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.8/download/linux-64/pytorch3d-0.7.8-py310_cu121_pyt241.tar.bz2
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

## Inference with the Pretrained Model

The pretrained model checkpoints are available in `experiment_data/checkpoints/pretrained_model`. Example meshes for evaluation are provided in `experiment_data/test_mesh/samples`.

Run the following command to perform inference:

```bash
python main.py --config-name test_pretrained
```

The generated results are saved under `experiment_data/output`:

- `optimized_seeds`: final seed positions optimized by the neural network.
- `remesh`: meshes extracted from the optimized seeds.

Inference settings can be configured in `config/test_pretrained.yaml`.

## Training a New Model

The complete configuration for data preparation, training, and testing is provided in `config/default.yaml`.

### Generate Training Data

The model is trained on the [Thingi10K dataset](https://ten-thousand-models.appspot.com). You can download the datasetby clicking **Download** on the website and using Google Drive.

Before generating the training data, update the following paths under `preprocessing` in `config/default.yaml`:

- `raw_dataset_dir`: path to the raw dataset.
- `training_file_dir`: directory in which the generated training files will be stored.

Generate the training data with:

```bash
python main.py mode=data_gen --config-name default
```

### Create Filelists

Set the output directory through `preprocessing.filelist_dir` in `config/default.yaml`, then run:

```bash
python main.py mode=split_dataset --config-name default
```

This command generates the training and evaluation filelists. The filelists used in our experiments are provided in `experiment_data/filelists/default`. Although the split was generated randomly, all network tuning experiments were based on these filelists. Reusing them may produce results that are more consistent with our experiments. To reuse them directly, set `preprocessing.filelist_dir` to this directory.

### Train

Run the following training stages in order:

```bash
python main.py stage=1 --config-name default
python main.py stage=2 --config-name default
python main.py stage=3 --config-name default
```

> **Note:** Training may not be fully reproducible under different hardware or environment configurations. If you observe clear non-convergence during Stage 1, please adjust the hyperparameters according to your results, such as increasing the regularization loss weight in `train.loss.weight` of the configuration file.

Training curves can be monitored with:

```bash
tensorboard --logdir experiment_data/logs
```

Model checkpoints are saved in `experiment_data/checkpoints`.

### Test

After training is complete, run inference with:

```bash
python main.py mode=test --config-name default
```

## Citation
```
@article{Xu2026NCVT,
  title   = {Neural Centroidal Voronoi Tessellations},
  author  = {Jiacheng Xu and Bo Pang and Rui Xu and Xiaocheng Zhang and Yang Liu and Fei Zhu and Guoping Wang and Peng-Shuai Wang},
  journal = {ACM Trans. Graph. (SIGGRAPH ASIA)},
  year    = {2026}
}
```
