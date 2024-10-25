import os
import json
import numpy as np
import pandas as pd

root_dir = "data/MOT17" if not os.getenv("ROOT_DIR") else os.getenv("ROOT_DIR")
ann_file = "train_cocoformat.json" if not os.getenv("ANN_FILE") else os.getenv("ANN_FILE")
out_file = "trackformer.json" if not os.getenv("OUT_FILE") else os.getenv("OUT_FILE")

with open(root_dir + "/" + ann_file, "r") as f:
    content = json.load(f)

categories = content['categories']
videos = content['videos']
images = content['images']
annotations = content['annotations']

df_videos = pd.DataFrame(videos)
df_images = pd.DataFrame(images)
df_annotations = pd.DataFrame(annotations)

trackformer_annotations = {
    'type': 'instances',
    'sequences': [],
    'images': [],
    'categories': list(map(lambda c: {"supercategory": c['name'], 'name': c['name'], 'id': int(c['id']) + 1}, categories)),
    'annotations': [],
    'frame_range': dict(start=0.0, end=1.0)
}

first_image_id = 0
next_annotation_id = 0
for _, video in df_videos.iterrows():
    video_images = df_images[df_images['video_id'] == video['id']]
    video_length = int(video_images.shape[0])

    trackformer_annotations['sequences'].append(video['name'])

    for i, (_, image) in enumerate(video_images.iterrows()):
        image_id = int(first_image_id + i)

        trackformer_annotations['images'].append({
            'id': image_id,
            'first_frame_image_id': first_image_id,
            'height': int(image['height']),
            'width': int(image['width']),
            'file_name': str(image['file_name']),
            'seq_length': video_length,
            'frame_id': i,
        })

        image_annotations = df_annotations[df_annotations['image_id'] == image['id']]
        unique_mot_instances = image_annotations['mot_instance_id'].unique()

        for _, annotation in image_annotations.iterrows():
            trackformer_annotations['annotations'].append({
                'id': next_annotation_id,
                'bbox': [int(annotation['bbox'][0]), int(annotation['bbox'][1]), int(annotation['bbox'][2]), int(annotation['bbox'][3])],
                'image_id': image_id,
                'segmentation': [],
                'ignore': 0,
                'visibility': 1.0,
                'area': float(annotation['area']),
                'iscrowd': 0,
                'category_id': int(annotation['category_id']) + 1,
                'seq': str(video['name']),
                'track_id': annotation['mot_instance_id'],
            })

            next_annotation_id += 1

    first_image_id += video_length


with open(out_file, 'w+') as f:
    print("Writing file ...")
    json.dump(trackformer_annotations, f, indent=4)