# Object Tracking

When you have trained a model and performed inference on the dataset you can use object tracking algorithms to detect object that appear in multiple frames of a stack. In this article we show you how this can be done using the CLI.

## Naive Object Tracking

The naive tracking algorithm uses the intersection over maximum metric to classify two bounding boxes of two different frames as the same object. You can run the naive tracking algorithm using the following command:

```bash
python cli.py naive-tracking \
    --dataset spine_mixed_train \
    --detections detections/faster_rcnn/faster_rcnn_run1_spine_mixed_train.csv \
    --output track.csv
```

Parameters:
`dataset` is one of the datasets defined in the `dataset.yml` configuration file.
`detections` is a path to a CSV file containing detected bounding boxes.
`output` is the path where the results of the naive tracking algorithm should be stored.

The resulting output file will look like this:

```csv
object_id,frame,cx,cy,w,h,score,filename,stack_id
0,0,382,19,19,22,0.7963060736656189,./datasets/spine/train\aidv853_date220321_tp1_stack3_sub11_layer116.png,0
1,0,356,42,18,17,0.7658138871192932,./datasets/spine/train\aidv853_date220321_tp1_stack3_sub11_layer116.png,0
2,0,390,54,18,18,0.6798791885375977,./datasets/spine/train\aidv853_date220321_tp1_stack3_sub11_layer116.png,0
0,0,21,104,17,19,0.6933675408363342,./datasets/spine/train\aidv853_date220321_tp1_stack3_sub12_layer069.png,2
1,0,37,89,17,15,0.3616678714752197,./datasets/spine/train\aidv853_date220321_tp1_stack3_sub12_layer069.png,2
0,0,479,274,19,18,0.4810413718223572,./datasets/spine/train\aidv853_date220321_tp2_stack2_sub21_layer063.png,3
1,0,163,346,16,15,0.2456881999969482,./datasets/spine/train\aidv853_date220321_tp2_stack2_sub21_layer063.png,3
```

The CSV file contains the tracking results for all stacks of the dataset. The stack ID is stored in the column `stack_id`. Within each stack ID each object that is considered to appear within the stack is assigned an object ID. The same stack id and object id can only occur once in a single frame.
