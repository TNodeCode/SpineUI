# Tracking Models

conda create -n mmtracking python==3.7.16 pip
conda activate mmtracking

```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

## Install mmdet

```bash
pip install -U openmim
mim install mmengine==0.10.4 mmcv-full==1.6.2 mmdet==2.28.2 mmtrack==0.14.0
pip install yapf==0.40.1
pip install git+https://github.com/JonathonLuiten/TrackEval.git
pip install -v -e .
```

## Clone Repository

```bash
git clone https://github.com/open-mmlab/mmtracking.git
```

## Verify installation

### with CUDA

```bash
python demo/demo_mot_vis.py configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py --input demo/demo.mp4 --output mot.mp4
```

### without CUDA

```
python demo/demo_mot_vis.py configs/mot/deepsort/sort_faster-rcnn_fpn_4e_mot17-private.py --input demo/demo.mp4 --output mot.mp4 --device cpu
```

## Train MOT model

### Convert dataset from MOT to COCO

```bash
python ./tools/convert_datasets/mot/mot2coco.py -i ./data/MOT17/ -o ./data/MOT17/annotations --split-train --convert-det
```

### Training

```bash
python tools/train.py .\configs\det\faster-rcnn_r50_fpn_4e_mot17.py
```

## Train REID model

### Convert dataset from MOT to REID

```bash
python ./tools/convert_datasets/mot/mot2reid.py -i ./data/MOT17/ -o ./data/MOT17/reid --val-split 0.2 --vis-threshold 0.3
```

### Training

```bash
python .\tools\train.py .\configs\reid\resnet50_b32x8_MOT17.py
```

## SORT, DeepSort, Tracktor Inference

Make a copy of one of the configuration file for the algorithm. You can find these configuration files in these directories:

First change the line where the checkpoint for the detector model is defined:

```python
model = dict(
    [...]
    detector=dict(
        [...]
        init_cfg=dict(
            [...]
            checkpoint='./work_dirs/faster-rcnn_r50_fpn_4e_mot17/epoch_4.pth'
```

Next change the line where the checkpoint for the REID model is defined:

```python
model = dict(
    [...]
    reid=dict(
        [...]
        init_cfg=dict(
            [...]
            checkpoint='./work_dirs/resnet50_b32x8_MOT17/epoch_6.pth'
```

This will make the tool use your own detector and REID model instead of downloading a pretrained model for inference.

Now use this config file for inference with the following command:

```bash
python ./demo/demo_mot_vis.py ./configs/mot/<algorithm_name>/<your_config_file>.py --input ./data/MOT17/train/stack1/img --output mydemo.mp4 --fps 1 --device cpu
```

## TrackEval Usage

### Ground Truths

Within your project there should be a directories named `data/MOT17/<train|test>` with the following structure:

```
data/MOT17
|- <train|test>
|-|- <stack_name_1>
|-|-|- gt
|-|-|-|- gt.txt
|-|-|- img
|-|-|-|- 000001.png
|-|-|-|- 000002.png
|-|-|-|- [...]
|-|-|- seqinfo.ini
|-|- <stack_name_2>
|-|-|- [...]
|-|- <stack_name_3>
|-|-|- [...]
|-|- seqmaps.txt
```

The file `seqmaps.txt` contains the names of all stacks in the following CSV format:

```csv
name
<stack_name_1>
<stack_name_2>
[...]
```

The first line is the header line and then for each stack a new line containing only the name of the stack is added. The stack names are the same as the directory names within `<train|test>`.

The file `gt.txt` contains the ground truth in the following format:

`<frame>,<object_id>,<x0>,<y0>,<width>,<height>,<confidence>,<x>,<y>,<z>`

Here is an example of how this could look like:

```
1,3,230,479,13,11,1.0,-1,-1,-1
2,3,230,478,15,17,1.0,-1,-1,-1
2,2,225,435,14,16,1.0,-1,-1,-1
```

You can find more about the MOT17 format here: https://motchallenge.net/instructions/

The file `seqinfo.ini` contains the info about the stack in the following format:

```
[Sequence]
name=<stack_name>
imDir=img
frameRate=1
seqLen=20
imWidth=512
imHeight=512
imExt=.png
seqLength=20
```

### Detections

When you want to evaluate a tracking model using metrics like MOTA or HOTA create a directory named `results/<model_name>` with the following structure:

```
results/
|- <model_name>
|-|- <seqmaps>
|-|-|- <stack_name_1>.txt
|-|-|- <stack_name_2>.txt
|-|-|- [...]
```

For each stack a text file with name `<stack_name>.txt` is created in the directory `results/<model_name>/seqmaps` is created with the same format as the ground truth data. Here is an example how the content of these files could look like:

```

1,3,232,481,14,9,0.65,-1,-1,-1
2,3,233,475,15,18,0.89,-1,-1,-1
2,2,223,431,16,16,0.71,-1,-1,-1
```

The values of `x0`, `y0`. `height`, `width` and `confidence` are those that were predicted by the model. If you have multiple models you can create a directory in `results` for each model.
