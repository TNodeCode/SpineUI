# Datasets

Managing datasets with SpineUI is straight forward. The configuration for your datasets is stored in `config/datasets.yml`.

## How it works

First of all you need to place all your images somewhere on your local disk. It is recommended to use the `./datasets` directory in this project, but you could also choose any other directory. Next you need to adapt the configuration file. In the configuration file datasets look like this:

```yaml
datasets:
  - name: my_dataset_train
    paths:
      - ./datasets/my_dataset/train/*.png
  - name: my_dataset_test
    paths:
      - ./datasets/my_dataset/test/*.png
```

For every dataset you need to add at least one entry to the `datasets` list. Every dataset needs to have a unique `name` and a list of `paths` where the images of that dataset are located. You can use glob expressions for images containing wildcards like `/path/image_day01_*.png`. With that syntax image directories can be used for multiple datasets. Also multiple paths can be added to a single dataset.

## Dividing Datasets into Stacks

### Using regular expressions to define stacks

In the Dentritic Spine Detection Project we have to deal with 3D information. When pictures are taken with the two-photon microscope multiple images of the same 3D voume are taken. It makes sense to use a naming schema for those images, so that images of the same 3D volume can be linked by their name. For example let's assume that we give each stack a unique numeric ID and name the images as folows: `stack_<stack_id>_<image_id>.png`.

Say that we have images of two stacks each containing three photos. So our dataset would have the following fienames:

```
stack_001_001.png
stack_001_002.png
stack_001_003.png
stack_002_001.png
stack_002_002.png
stack_002_003.png
```

For each stack we need to create an entry of the `datasets.stacks` object in the configuration file. This object will have the keys `regex` and `stack_name`.

`regex` contains a regular expression that is applied to all images that are found in the dataset paths. If the regular expression matches a filename this image becomes part of the stack. An image can also be part of multiple stacks. Regular expressions should always start end end with the `.*` pattern, because otherwise some operating systems won't be able to match any filename. All variable parts of the filename shoud be replaced by groups, which are denoted with round brackets `()` in a regular expression. Within these brackets there is the pattern for the variable part of the filename. For example `[0-9]{1,3}` means that the numbers from zero to nine can appear at least one time and at most three times in this group.

```yaml
datasets:
  - name: spine_tracking
    stacks:
      patterns:
        - regex: .*stack_([0-9]{1,3})_([0-9]{1,3}).*
          stack_name: "Stack $1"
```

`stack_name` contains the final name of the stack which is used for grouping the images. Every `$i` part of that name is replaced by the value of the i-th group of the regular expression. So in the given example the stack names for the images would be:

```
Stack 001
Stack 001
Stack 001
Stack 002
Stack 002
Stack 002
```

So this naming scheme will result in two different stack names you can choose from in the UI. If we would change the stack naming schema to `stack_name: Stack $1 Image $2` we would end up with six different stacks each containing a single image.

### Multiple patterns

Now let's assume we add some more images to our dataset which have a different naming schema. The filenames of the new images look ike this:

```
S3I1
S3I2
S3I3
```

These filenames would not be matched by the pattern we defined before. What we can do is to add another pattern to our stack configuration like this:

```yaml
datasets:
  - name: spine_tracking
    stacks:
      patterns:
        - regex: .*stack_([0-9]{1,3})_([0-9]{1,3}).*
          stack_name: "Stack $1"
        - regex: .*S([0-9]{1,3})I([0-9]{1,3}).*
          stack_name: "Stack $1"
```

## Annotations

Now that we know how we can add datasets to our configuration and how to divide these datasets into stacks we need to add the annotations of our images to the dataset. SpineUI can handle multiple annotations for the same images, so that if for example multiple persons annotated the same data you can use SpineUI to compare those annotations.

Annotations are configured under the `datasets.[*].annotations` key of a dataset object in the configuration file.

### Adding bounding box annotations in the COCO format

If you want to add bounding boxes to your dataset in the COCO format add a new list item to the list in `datasets[*].annotations` with the following keys:

`name`: A name for the annotations which you can choose. This name will be displayed in the SpineUI app. 

`type`: The annotation type, `coco` in this case.

`paths`: The COCO JSON annotation file. This mist be passed as a list, although we usually only have a single JSON annotation file. This is because other formats like Masks require multiple annotation sources.

This is how bounding box annotations can look like in the configuration file.

```yaml
datasets:
  - name: spine_tracking
    annotations:
      - name: annotator01
        type: coco
        paths: 
          - ./datasets/spine/annotations/instances_train2017.json
```

### Adding mask annotations

SpineUI can also handle mask annotations. Mask annotations use black and white PNG images for storing the mask annotation. You need to place a PNG file with the same name as the dataset image in a directory and then add this directory name to `datasets[*].annotations.paths`. Mask annotation images can also be placed in multiple directories. Then you just need to add each path to the key mentioned above. Also the value of `datasets[*].annotations.type` must be set to `masks`.

An annotation for masks could look like this:

```yaml
datasets:
  - name: spine_tracking
    annotations:
      - name: mask-annotations-01
        type: masks
        paths: 
          - ./datasets/spine/train-sam-b/*
```

## Training Results

There are two types of annotations which are distinguished in SpineUI: ground truth annotations and the annotations generated by a model. Annotations generated by a model are stored in CSV files and are places in the configuration under `datasets[*].detections`. Each detection has the following entries:

`name`: The name that should be used for the detections when displayed in SpineUI. It makes sense to use the model name here.

`paths`: Paths to CSV files containing the detected bounding boxes.

A configuration for two training results could look like this:

```yaml
datasets:
  - name: spine_tracking
    detections:
      - name: Cascade RCNN
        paths:
          - ./detections/cascade_rcnn/cascade_rcnn_run*_spine_mixed_train.csv
      - name: Faster RCNN
        paths:
          - ./detections/faster_rcnn/faster_rcnn_run*_spine_mixed_train.csv
```

