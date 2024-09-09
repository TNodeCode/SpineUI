# CrowdHuman Dataset

CrowdHuman is a popular standard dataset for training MOT models. Many MOT models accept datasets in the CrowdHuman annotation format. We will explain this annotation format in this article.

## Tracking Annotations

Tracking ground truth data is stored in a ODGT format, which is basically a JSON file but with each line containing a JSON document instead of the whole file containing a single JSON document. If you want to read this annotation file with Python you have to split it into lines and then parse each line as a JSON document. This is the schema of the tracking annotation ground truth.

```
JSON{
    "ID" : image_filename,
    "gtboxes" : [gtbox], 
}

gtbox{
    "tag" : "person" or "mask", 
    "vbox": [x, y, w, h],
    "fbox": [x, y, w, h],
    "hbox": [x, y, w, h],
    "extra" : extra, 
    "head_attr" : head_attr, 
}

extra{
    "ignore": 0 or 1,
    "box_id": int,
    "occ": int,
}

head_attr{
    "ignore": 0 or 1,
    "unsure": int,
    "occ": int,
}
```

An example of these annotations could look like this:

```json
{
  "ID": "273271,c9db000d5146c15",
  "gtboxes": [
    {
      "fbox": [
        72,
        202,
        163,
        503
      ],
      "tag": "person",
      "hbox": [
        171,
        208,
        62,
        83
      ],
      "extra": {
        "box_id": 0,
        "occ": 0
      },
      "vbox": [
        72,
        202,
        163,
        398
      ],
      "head_attr": {
        "ignore": 0,
        "occ": 0,
        "unsure": 0
      }
    },
    ...
}
{
  "ID": "273271,c9db000d5146c15",
  "gtboxes": [
    ...
  ]
  ...
}
...
```

The filennames are denoted without file extensions.

You can find more about the annotation format here: https://www.crowdhuman.org/download.html or in the paper here: https://arxiv.org/pdf/1805.00123

There are three different bounding boxes for each object. The ***full body box*** (fbox) contains the complete object, whether it is partially hidden or not. You can see examples of full body boxes below. They are are denoted with continuous, thick lines. The ***visible region boxes*** (vbox) contain only the visible part of an object. Vboxes always have an intersection with fboxes of 100%. Third there are ***head bounding boxes*** (hbox) which contain the head of a person. They also have an inttersection of 100% with the corresponding fboxes.

<img src="https://www.crowdhuman.org/images/5.jpg" />
source: https://www.crowdhuman.org

### Transfering box format to other datasets

If you want to annotate your own dataset with this annotation format and your dataset doesn't contain persons but other objects and objects event aren't hidden just use teh same bounding box coordinates for all three bounding box types. You should also set the attribute `gtboxes.headattr.ignore` to `1`.