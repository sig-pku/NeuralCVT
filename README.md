# Neural CVT

## Environment Setup

The environment setup is based on Conda and has been tested on Linux servers without sudo access.

Create the conda environment:

```bash
conda create -n ncvt python=3.10 -y
conda activate ncvt
```

This repository uses [Git LFS](https://git-lfs.com/) to manage model checkpoint files. Install and initialize Git LFS before cloning:

```bash
conda install -c conda-forge git-lfs -y
git lfs install
```

Clone the repository together with all submodules:

```bash
git clone --recurse-submodules https://github.com/sig-pku/NeuralCVT.git
cd NeuralCVT
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

If a CMake environment is not already available, set up one and build Geogram using the following steps:

```bash
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

## Inference with Pretrained Models

Pretrained model checkpoints are available in `experiment_data/checkpoints/pretrained_model`, and example meshes for evaluation are provided in `experiment_data/test_mesh/samples`.

Run the following command to perform inference with the pretrained model:

```bash
python main.py --config-name test_pretrained
```

The generated results are saved under `experiment_data/output`:

- `optimized_seeds` contains the final optimized seed results produced by the neural network.
- `remesh` contains the extracted meshes.

Inference settings can be configured in `config/test_pretrained.yaml`.

## Train a New Model

The complete configuration for data preparation, training, and testing is provided in `config/default.yaml`.

### Generate Training Data

The model is trained on the [Thingi10K dataset](https://ten-thousand-models.appspot.com). You can download by clicking **Download** on the website and using Google Drive.

Before generating data, modify the following paths under `preprocessing` in `config/default.yaml`:

- `raw_dataset_dir`: local path to the raw dataset.
- `training_file_dir`: local path for generated training files.

Generate the training data with:

```bash
python main.py mode=data_gen --config-name default
```

### Create Filelists

Set the output directory for filelists through `preprocessing -> filelist_dir`, then run:

```bash
python main.py mode=split_dataset --config-name default
```

This command creates the train and evaluation filelists. The filelists used in our experiments are provided in `experiment_data/filelists/default`. Although the split is randomly generated, all network tuning experiments were based on these filelists; using them can produce results that are more consistent with our experiments. To reuse them directly, set `filelist_dir` to this directory without generating new filelists.

### Train

Run the following stages in order:

```bash
python main.py stage=1 --config-name default
python main.py stage=2 --config-name default
python main.py stage=3 --config-name default
```

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
