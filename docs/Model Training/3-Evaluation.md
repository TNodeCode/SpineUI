# Model Evaluation

In this artice we will show you how you can evaluate the models you have trained. Evaluation means that you get accuracy, precision, recall and F1-scores for each epoch of your training process.

## Evaluation

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