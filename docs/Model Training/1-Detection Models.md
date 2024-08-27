# Detection Models

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

## Model Training

In this article we will show you how you can use this repository for training models that can be used for SpineUI.

### Training

For training there is a CLI written with the Python Click framework which you can use. For simplicity you can create a Bash file that calls this API so that you do not have to type the whole command each time you want to train a model.

Create a file named `train.sh` with the following content in the directory where the CLI is located.

```bash
export CONFIG_DIR=configs
export DATASET_DIR=data/spine
export ANNOTATIONS_TRAIN=$DATASET_DIR/annotations/instances_train2017.json
export ANNOTATIONS_VAL=$DATASET_DIR/annotations/instances_val2017.json
export ANNOTATIONS_TEST=$DATASET_DIR/annotations/instances_test2017.json
export IMAGES_TRAIN=$DATASET_DIR/train2017
export IMAGES_VAL=$DATASET_DIR/val2017
export IMAGES_TEST=$DATASET_DIR/test2017
export CLASSES="classes.txt"
export WORKERS=4
export BATCH_SIZE=16
export MODEL_TYPE=faster_rcnn
export MODEL_NAME=faster_rcnn_r50_fpn_1x_coco
export EPOCHS=25
export WORK_DIR=runs/$MODEL_TYPE/$MODEL_NAME

python cli.py train \
    --config_dir $CONFIG_DIR \
    --train_annotations $ANNOTATIONS_TRAIN \
    --train_images $IMAGES_TRAIN \
    --val_annotations $ANNOTATIONS_VAL \
    --val_images $IMAGES_VAL \
    --test_annotations $ANNOTATIONS_TEST \
    --test_images $IMAGES_TEST \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --epochs $EPOCHS \
    --classes classes.txt \
    --batch_size $BATCH_SIZE \
    --work_dir $WORK_DIR
```

After training the results will be saved in `./runs/<model_name>/<submodel_name>`. You can find the model weights as `.pth` files for each epochs and the training log files there. Also there is a file named `<model_name>.py` which contains all the parameters used for training the model.

## Model Evaluation

In this artice we will show you how you can evaluate the models you have trained. Evaluation means that you get accuracy, precision, recall and F1-scores for each epoch of your training process.

### Evaluation

You can use the CLI again for evaluating your trained model. If you want to evaluate the model for each of the datasets (train / val / test) you can do this with the following command:

```bash
export CONFIG_DIR=configs

export DATASET_DIR=data/spine
export CLASSES="classes.txt"
export ANNOTATIONS_TRAIN=$DATASET_DIR/annotations/instances_train2017.json
export ANNOTATIONS_VAL=$DATASET_DIR/annotations/instances_val2017.json
export ANNOTATIONS_TEST=$DATASET_DIR/annotations/instances_test2017.json
export IMAGES_TRAIN=$DATASET_DIR/train2017
export IMAGES_VAL=$DATASET_DIR/val2017
export IMAGES_TEST=$DATASET_DIR/test2017

export MODEL_TYPE=faster_rcnn
export MODEL_NAME=faster_rcnn_x101_64x4d_fpn_1x_coco
export BATCH_SIZE=8
export EPOCHS=25
export WORK_DIR=runs/$MODEL_TYPE/$MODEL_NAME

# Evaluate detected bounding boxes for all datasets
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_TRAIN \
    --epochs $EPOCHS \
    --csv_file_pattern detections_train_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_train.csv

# Evaluate detected bounding boxes for validation dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_VAL \
    --epochs $EPOCHS \
    --csv_file_pattern detections_val_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_val.csv

# Evaluate detected bounding boxes for test dataset
python cli.py eval \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --annotations $ANNOTATIONS_TEST \
    --epochs $EPOCHS \
    --csv_file_pattern detections_test_epoch_\$i.csv \
    --results_file eval_${MODEL_NAME}_test.csv
```

This will create CSV files with names `eval_<model_name>_<dataset_name>.csv` in the directory `./runs/<model_name>/<submodel_name>`. The contents of this CSV will look like this:

```csv
ap,ar,f1,ap_50,ap_75,ap_small,ap_medium,ap_large,ar_1,ar_10,ar_100,ar_small,ar_medium,ar_large
0.018238665971860343,0.018142235123367198,0.018190322748297905,0.018238665971860343,0.018238665971860343,0.0177267288132322,0.0,,0.011611030478955007,0.0171988388969521,0.0171988388969521,0.017362637362637365,0.0,
0.5607005189383254,0.6262699564586357,0.5916741770083266,0.5607005189383254,0.5214491508295125,0.45215635099911644,0.6160537482319663,,0.11589259796806968,0.4957910014513788,0.5521770682148041,0.5517948717948717,0.6461538461538461,
0.6702040721260409,0.762699564586357,0.7134664758997078,0.6702040721260409,0.6223362383788008,0.5603092855567239,0.5349535368283374,,0.12017416545718436,0.5731494920174166,0.6701015965166909,0.6706227106227107,0.6538461538461539,

```

It contains the following metrics for each epoch:

- Precision
- Recall
- F1-Score
- AP50 score
- AP75 score
- AP small score
- AP medium score
- AP large score
- AR1 score
- AR10 score
- AR100 score
- AR small score
- AR medium score
- AR large score

## Inference

In this article we will show you how you can evaluate your trained model.

### Predict Bounding Boxes

Like for training you can also use the CLI for evaluating the model. We assume that you trained the model and that there are `.pth` files for each epoch in the directory `./runs/<model_name>/<submodel_name>`.

The first step is to use the model for predicting bounding boxes for all images in the dataset. If you want to predict the bounding boxes for a given epoch and one of the three datasets (train / val / test) you can do this with the following command:

```bash
# Detect bounding boxes for the first epoch of the training dataset
export DATASET_DIR=data/spine
export DATASET=train
export MODEL_TYPE=faster_rcnn
export MODEL_NAME=faster_rcnn_x101_64x4d_fpn_1x_coco
export BATCH_SIZE=8
export EPOCH=1


$ python cli.py detect \
    --model_type $MODEL_TYPE \
    --model_name $MODEL_NAME \
    --weight_file epoch_$EPOCH.pth \
    --image_files $DATASET_DIR/$DATASET"'2017/*.png'" \
    --results_file detections_${DATASET}_epoch_${EPOCH}.csv \
    --batch_size $BATCH_SIZE \
    --device cuda:0
```

If you want to get predictions for all epochs and all datasets you can modify the bash script in the following way:

```bash
# Detect bounding boxes for all datasets and epochs
export DATASET_DIR=data/spine
export MODEL_TYPE=faster_rcnn
export MODEL_NAME=faster_rcnn_x101_64x4d_fpn_1x_coco
export BATCH_SIZE=8

for DATASET in "train" "val" "test"
do
    for EPOCH in $(seq 1 $EPOCHS)
    do
        python cli.py detect \
            --model_type $MODEL_TYPE \
            --model_name $MODEL_NAME \
            --weight_file epoch_$EPOCH.pth \
            --image_files $DATASET_DIR/$DATASET"'2017/*.png'" \
            --results_file detections_${DATASET}_epoch_${EPOCH}.csv \
            --batch_size $BATCH_SIZE \
            --device cuda:0
    done
done
```

When running the `detect` command a CSV file containing the bounding boxes will be created for each dataset and epoch. You can find these CSV files in the directory `./runs/<model_name>/<submodel_name>`.

The content of the CSV file will look like this:

```csv
filename,class_index,class_name,xmin,ymin,xmax,ymax,score
aidv853_date220321_tp1_stack1_sub22_layer081.png,0,spine,161,400,180,420,0.16251738369464874
aidv853_date220321_tp1_stack1_sub22_layer081.png,0,spine,499,87,512,107,0.14402839541435242
aidv853_date220321_tp2_stack0_sub22_layer025.png,0,spine,38,82,59,97,0.2981097996234894
```

It contains the following information:

- Filename
- Numeric class index and class name of the bounding box
- Coordinates in XYXY format
- Confidence Score

The detection CSV files can now be used in the SpineUI app for visually showing the detected bounding boxes of the model.
