# Trackformer Annotations

Sample annotation file

```json
{
    "type": "instances",
    "categories": [
        {
            "supercategory": "spine",
            "name": "spine",
            "id": 1
        },
        ...
    ],
    "images": [
        {
            "file_name": "aid052N1D1_tp1_stack2_layer001.png",
            "height": 512,
            "width": 512,
            "id": 0,
            "first_frame_image_id": 0,
            "seq_length": 20,
            "frame_id": 0
        },
        ...
    ],
    "annotations": [
        {
            "id": 0,
            "category_id": 1,
            "image_id": 3,
            "seq": "aid052N1D1_tp1_stack2",
            "track_id": 0
            "bbox": [
                337,
                473,
                23,
                21
            ],
            "area": 483,
            "segmentation": [],
            "ignore": 0,
            "visibility": 1.0,
            "iscrowd": 0,
        },
        ...
    ],
    "sequences": [
        "aid052N1D1_tp1_stack2",
        ...
    ],
    "frame_range": {
        "start": 0.0,
        "end": 1.0
    }
}
```

## JSON annotation file structure
```mermaid
erDiagram
    DATASET {
        string type "Dataset type ('instances' for tracking)"
        sequences list[string] "List of sequence names"
        frame_range object "Object describing frame range"
    }
    CATEGORY }o--|| DATASET : has
    IMAGE }o--|| DATASET : has
    ANNOTATION }o--|| DATASET : has
    ANNOTATION }|--|| IMAGE : has
    ANNOTATION }|--|| CATEGORY : has
```

## Category entities
``` mermaid
erDiagram
    CATEGORY {
        int id PK "The category ID starting from 1"
        string supercategory "Name of the supercategory (use the same name as for 'name')"
        string name "The category name"
    }
```

## Image entities
```mermaid
erDiagram
    IMAGE {
        id int PK "Image ID starting from 1"
        file_name string "Image filename (relative to train directory)"
        height int "Image height"
        width int "Image width"
        frame_id int "Frame ID (starting from 0)"
        first_frame_image_id int "ID of the first frame in the sequence"
        seq_length int "Number of frames in the corresponding sequence"
    }
```

## Annotation entities
```mermaid
erDiagram
    ANNOTATION {
        id int PK "Annotation ID (starting from 1)"
        category_id int FK "ID of the category of this bounding box"
        image_id int FK "ID of the image"
        seq string "Sequence name"
        track_id int "ID of track in the sequence (starting from 0)"
        bbox list[int] "Bounding box in XYWH format"
        segmentation list[int] "Segmentation mask polygon"
        area int "Bounding box area (width * height)"
        iscrowd int "Always 0"
        ignore int "Always 0"
        visibility float "Object visibility"
    }
```