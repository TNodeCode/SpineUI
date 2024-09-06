# MMDetection Tracking Dataset

Sample annotation file

```json
{
    "categories": [
        {
            "id": 1,
            "name": "pedestrian"
        }
    ],
    "videos": [
        {
            "id": 1,
            "name": "aid052N1D3_tp1_stack1_default_aug_False_epoch_19_theta_0.5_delta_0.1_Test",
            "fps": 1,
            "width": 512,
            "height": 512
        },
        ...
    ],
    "images": [
        {
            "id": 1,
            "video_id": 1,
            "file_name": "aid052N1D3_tp1_stack1_default_aug_False_epoch_19_theta_0.5_delta_0.1_Test\\img\\000001.png",
            "height": 512,
            "width": 512,
            "frame_id": 0,
            "mot_frame_id": 1
        },
        ...
    ],
    "annotations": [
        {
            "category_id": 1,
            "bbox": [
                22.0,
                129.0,
                18.0,
                14.0
            ],
            "area": 252.0,
            "iscrowd": false,
            "visibility": 1.0,
            "mot_instance_id": 0,
            "mot_conf": 1.0,
            "mot_class_id": 0,
            "id": 1,
            "image_id": 1,
            "instance_id": 0
        },
        ...
    ],
    "num_instances": 784
}
```

## JSON annotation file structure
```mermaid
erDiagram
    DATASET {
        int num_instances "Number of distinct annotation instance ids"
    }
    CATEGORY }o--|| DATASET : has
    VIDEO }o--|| DATASET : has
    IMAGE }o--|| DATASET : has
    ANNOTATION }o--|| DATASET : has
```

## Category entities
``` mermaid
erDiagram
    CATEGORY {
        int id PK "The category ID starting from 1"
        string name "The category name"
    }
```

## Video entities
``` mermaid
erDiagram
    VIDEO {
        int id PK "The sequence / video ID"
        string name "The sequence / video name"
        fps int "frames per second of the video"
        width int "frame / image width"
        height int "frame / image height"
    }
```

## Image entities
```mermaid
erDiagram
    IMAGE {
        id int PK "Image ID starting from 1"
        video_id int "FK Video ID"
        file_name string "Image filename (relative to train directory)"
        height int "Image height"
        width int "Image width"
        frame_id int "Frame ID (starting from 0)"
        mot_frame_id int "MOT frame ID (starting from 1)"
    }
```

## Annotation entities
```mermaid
erDiagram
    ANNOTATION {
        id int PK "Annotation ID (starting from 1)"
        category_id int FK "ID of the category of this bounding box"
        bbox list[int] "Bounding box in XYXY format"
        area int "Bounding box area (width * height)"
        iscrowd bool "Always false"
        visibility float "Always 1.0"
        mot_instance_id int "Instance ID of this object (every distinct object has its own ID)"
        mot_conf float "Confidence score (always 1.0 in GT)"
        mot_class_id int "Class ID (same as category_id minus 1)"
        instance_id int "Instance ID"
    }
```