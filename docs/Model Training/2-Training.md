# Model Training

In this article we will show you how you can use this repository for training models that can be used for SpineUI.

## Training

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

