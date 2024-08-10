# Setup of the training environment

In this article we go through the steps for installing the neccessary libraries for training a model.

## Setup

Before you can train the models you need to install the necessary libraries. We assume you have already created a Conda environment. If not here is how you can create a new one:

```bash
$ conda create -n spine python==3.11 pip
$ conda activate spine
$ python -m pip install --upgrade pip
```

For the training process we need to install PyTorch with CUDA support. Run this command to install PyTorch with CUDA support:

```bash
$ python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

The next ibrary we need to install is the MMCv library. The MMCV library is already included in this repository and should be installed with the following commands. It should always be built from scrat with the commands shown and not be installed by downloading it from PyPi, otherwise the version installed might not fit your platform. Make sure a C++ compiler and NVCC is available on your machine and a CUDA device is available, otherwise the installation will fail.

```bash
$ cd mmcv
$ FORCE_CUDA=1 MMCV_WITH_OPS=1 pip install -e . -v
```

The next step is to install the needed Python libraries.

```bash
$ python -m pip install cmake
$ python -m pip install -i https://test.pypi.org/simple/ tnc-process
$ python -m pip install -r requirements/build.txt
$ python -m pip install -r requirements/optional.txt
$ python -m pip install -r requirements/runtime.txt
```