# Model Evaluation

In this article we will show you how you can evaluate your trained model.

## Predict Bounding Boxes

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