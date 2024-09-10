# Object Tracking

When you have trained a model and performed inference on the dataset you can use object tracking algorithms to detect object that appear in multiple frames of a stack. In this article we show you how this can be done using the CLI.

## Naive Object Tracking

The naive tracking algorithm uses the intersection over maximum metric to classify two bounding boxes of two different frames as the same object. You can run the naive tracking algorithm using the following command:

```bash
python cli.py naive-tracking \
    --dataset spine_mixed_train \
    --detections detections/faster_rcnn/faster_rcnn_run1_spine_mixed_train.csv \
    --output-dir detections/naive-tracking \
    --threshold 0.5
```

Parameters:<br>
`dataset` is one of the datasets defined in the `dataset.yml` configuration file.<br>
`detections` is a path to a CSV file containing detected bounding boxes.<br>
`output-dir` is the path where the results of the naive tracking algorithm should be stored.<br>
`threshold` is the minimum confidence score that should be reached for a bounding box<br>

The resulting output file contains the tracking results in the MOT17 format and will look like this:

```csv
1,0,33,196,13,14,0.987898051738739,-1,-1,-1
2,0,34,196,14,14,0.9786934852600098,-1,-1,-1
3,0,32,193,15,15,0.8812576532363892,-1,-1,-1
4,0,34,196,14,14,0.9571372270584106,-1,-1,-1
1,1,22,130,18,15,0.9855337738990784,-1,-1,-1
```

Each line represents a detected bounding box in the following format:
`<frame>,<object_id>,<x0>,<y0>,<w>,<h>,<score>,<x>,<y>,<z>`

## Tracking Evaluation

After the tracking CSV file is generated you can evaluate the tracking results agains the ground truth data with the following command:

```bash
python cli.py eval-tracking \
    --gt-folder datasets/MOT17/train \
    --detections detections/naive-tracking \
    --output-dir detections/naive-tracking-evaluation \
    --similarity-metric IoM
```

Parameters:<br>
`--gt-folder`: Ground truth data in MOT17 format<br>
`--detections`: Directory that contains the tracking result CSV files (one per stack)<br>
`--output-dir`: Directory where evaluation results should be stored<br>
`--similarity-metric`: Metric for computing similarity of bounding boxes (IoU or IoM)
