# Trackformer

Trackformer is an end-to-end object tracking model. You can find the source code inside the `trackformer` directory of this project.

You cn find the source code we used for training the trackformer model in the `repositories/trackformer` directory of this Git repository.

## Environment

Trackformer was built to run in an environment where Python 3.7.x should be installed. There are several options for initializing such an environment on your system. We will show you two ways how you can setup this environment on your machine using Conda or Docker.


## Setup with WSL 

You can setup Trackformer in a WSL environment under Windows. We have a separate article in this documentation about installing Linux CUDA environments under Windows. Follow the steps of this article and then follow the installation process described below for Conda environments.


## Setup with Docker on Windows

It is possible to run Trackformer inside a Docker container on Windows. We will explain the stept for creating the Docker image and how to run this Docker image in the following section.

## Spine dataset

Place the Spine MOT dataset in this directory: `data/spine`. The directory structure should now look like this:

```
data
|- spine
  |- annotations
    |- test.json
    |- train.json
    |- val.json
  |- test
    |- <stack_name>
      |- det
      |- gt
      |- img
      |- seqinfo.ini
    |- <stack_name>
      |- ...
  |- train
    |- <stack_name>
      |- det
      |- gt
      |- img
      |- seqinfo.ini
    |- <stack_name>
      |- ...
  |- val
    |- <stack_name>
      |- det
      |- gt
      |- img
      |- seqinfo.ini
    |- <stack_name>
      |- ...
```

### Build the Docker image

First we need to build the docker image. We have included a Dockerfile inside this repository under `trackformer/Dockerfile`. For building the image run the following command:

```bash
$ docker build -t tnodecode/trackformer-base
```

This process can take a while because the base image used is very large. After the command has finished you can now create an instance of this docker image and open its shell with the following command:

```bash
docker run \
    -it \
    --gpus all \
    --shm-size=16gb \
    --rm \
    tnodecode/trackformer-base \
    bash
```

Your terminal should now look like this:

```bash
root@3139f5fc0c8d:/app#
```

This means you are using the shell of your docker container. It is similar to connecting to a remote machine via SSH.

Now we have created a Docker image that contains a PyTorch runtime together with the CUDA development tools which are needed for compiling the CUDA operators provided in `src/trackformer/models/ops`. Unfortunately Docker cannot access the GPU during image built time, so we have install the CUDA runtime and compie the operators via the Docker container shell. 

In the docker container shell run the following two commands and DO NOT CLOSE the shell afterwards.

```bash
# First switch to the conda trackformer environment inside the container
$ conda init
$ source ~/.bashrc
$ conda activate trackformer
# Install NVIDIA CUDA toolkit
$ apt-get install -y nvidia-cuda-dev nvidia-cuda-toolkit
# Compile the operators
$ python src/trackformer/models/ops/setup.py build --build-base=src/trackformer/models/ops/ install
# Update numpy, scipy and pandas
$ pip install -U numpy scipy pandas
```

When both commands are executed open a second terminal while the docker shell is still opened. In the second terminal type `docker ps` to get a list of all running docker containers. Find the container that you started and its name (`clever_bhabha` in this case).

```
$ docker ps
CONTAINER ID   IMAGE                   COMMAND   CREATED         STATUS         PORTS     NAMES
3139f5fc0c8d   tnodecode/trackformer-base   "bash"    2 minutes ago   Up 2 minutes             clever_bhabha
```

Now we want to create a new docker image based on our currently running container. We can do this with the following command:

```bash
$ docker commit clever_bhabha tnodecode/trackformer
```

This will create a new docker image with tag `tnodecode/trackformer` based on the container with the name `clever_bhabha`. After this command you can close the docker container shell in the first terminal by simply typing "exit" and pressing enter.

Now you can create a container that is based on our new image by executing the following command:

```bash
$ docker run \
    -it \
    --gpus all \
    --shm-size=16gb \
    --rm \
    tnodecode/trackformer \
    bash
```

### Train model on Windows with Docker

Now that we have built the image we can start training a Trackformer model. First download the pretrained models and place them inside the `models` directory. Then place the dataset inside the `data/spine` directory. Also create a directory named `checkpoints` where trackformer will store the ResNet50 or ResNet101 pretrained models.

Now you can run the following command:

```bash
docker run \
    --gpus all \
    --shm-size=16gb \
    --rm \
    -v $PWD/cfgs:/app/cfgs \
    -v $PWD/data:/app/data \
    -v $PWD/models:/app/models \
    -v $PWD/src:/app/src \
    -v $PWD/checkpoints:/root/.cache/torch/hub/checkpoints \
    tnodecode/trackformer \
    conda run --no-capture-output -n trackformer \
    python src/train.py with \
    mot17 \
    deformable \
    multi_frame \
    tracking \
    device=cuda:0 \
    output_dir=checkpoints/custom_dataset_deformable \
    mot_path_train=data/spine \
    mot_path_val=data/spine \
    train_split=train \
    val_split=val \
    epochs=20 \
```

The option `-v $PWD/cfgs:/app/cfgs` maps a directory from Windows into the container. PWD is the absolute path of your project (C:/Users/<username>/<project_dir>). We do this so that we can make changes to the code or the dataset without rebuilding the docker image. Also the created model weights are stored on our Windows filesystem instead on the container filesystem that will be destroyed when the container is shut down. The directory `/root/.cache/torch/hub/checkpoints` inside the container stores the downloaded models from Torch Hub. If we wouldn't map this directory to the `checkpoints` directory of our project the conainer woud download the model each time we restart the container because its filesystem is not persisted on the disk.

### Train a model on a Linux machine

Training the model on a linux machine can be done by the following command:

```bash
python src/train.py with \
    mot17 \
    deformable \
    multi_frame \
    tracking \
    device=cuda:0 \
    output_dir=checkpoints/custom_dataset_deformable \
    mot_path_train=data/spine \
    mot_path_val=data/spine \
    train_split=train \
    val_split=val \
    epochs=20 \
```

We also created a script named `train_custom.sh` in the root directory of the trackformer repository which contains this command.


### Inference

Inference for all stacks of a dataset for all splits can be done with the following bash script:

```bash
export DATASET=spine        # dataset name
export SUBDIR=run_1         # subdir in checkpoints dir

for MODEL_NAME in MOTA IDF1 BBOX_AP_IoU_0_50
do
    for SPLIT in train val test
    do
        for file in ./data/$DATASET/$SPLIT/*
        do
        if [ -d "$file" ]; then
            STACK_NAME="$(basename -- $file)"
            echo "Processing" $SPLIT $STACK_NAME "..."
            python src/track.py with \
                dataset_name=$STACK_NAME \
                obj_detect_checkpoint_file=checkpoints/$SUBDIR/checkpoint_best_$MODEL_NAME.pth \
                write_images=True \
                generate_attention_maps=False \
                output_dir=detections/$SUBDIR/$MODEL_NAME/$SPLIT
        fi
        done
    done
done
```

This will run inference on all images found in `data/spine/$SPLIT` and save renderen images with bounding boxes and trackformer detection files in `detections/$SUBDIR/$MODEL_NAME/$SPLIT`.

The repository also contains this bash code in the file `track_custom.sh` in the root directory.