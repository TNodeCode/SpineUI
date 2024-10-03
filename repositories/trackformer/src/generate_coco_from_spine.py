import json
import os
import cv2
import pandas as pd
import numpy as np
 
DATA_ROOT = "data/" + os.getenv('DATASET') if os.getenv('DATASET') else "data/spine_dataset"
VIS_THRESHOLD = 0.0


def generate_coco_from_spine(split_name='train_val', split='train_val'):
    """
    Generate COCO data from spine dataset.
    """
    annotations = {}
    annotations['type'] = 'instances'
    annotations['images'] = []
    annotations['categories'] = [{"supercategory": "spine",
                                  "name": "spine",
                                  "id": 1}]
    annotations['annotations'] = []
    annotation_file = os.path.join(DATA_ROOT, f'annotations/{split_name}.json')

    # IMAGES
    imgs_list_dir = os.listdir(os.path.join(DATA_ROOT, split))
    seqs_names = sorted(list(set([img.split('_layer')[0] for img in imgs_list_dir])))
    seqs_lengths = [len([img for img in imgs_list_dir if seq in img]) for seq in seqs_names]
    first_frame_image_id = 0
    for i, img in enumerate(sorted(imgs_list_dir)):
        im = cv2.imread(os.path.join(DATA_ROOT, split, img))
        h, w, _ = im.shape
        seq_name = img.split('_layer')[0]
        if i > 0 and seq_name != imgs_list_dir[i-1].split('_layer')[0]:
            first_frame_image_id = i

        annotations['images'].append({
            "file_name": img,
            "height": h,
            "width": w,
            "id": i, 
            "first_frame_image_id": first_frame_image_id,
            "seq_length": seqs_lengths[seqs_names.index(seq_name)],
            "frame_id": i - first_frame_image_id,
        })

    # GT
    annotation_id = 0
    img_file_name_to_id = {
        os.path.splitext(img_dict['file_name'])[0]: img_dict['id']
        for img_dict in annotations['images']}

    for split in ['train', 'val', 'test']:
        if split not in split_name:
            continue
        annos_file = os.path.join(DATA_ROOT, f'annotations/{split}.csv')
        df = pd.read_csv(annos_file)

        xmins = np.around(df['xmin']).astype(int).values
        ymins = np.around(df['ymin'].values).astype(int)
        xmaxs = np.around(df['xmax'].values).astype(int)
        ymaxs = np.around(df['ymax'].values).astype(int)
        bbox_widths = xmaxs - xmins
        bbox_heights = ymaxs - ymins
        areas = (bbox_widths * bbox_heights).astype(int).tolist()
        track_ids = df['id'].values

        gtboxes = np.array([xmins, ymins, bbox_widths, bbox_heights]).T # (x, y, width, height)
        for i in range(len(gtboxes)):
            visibility = 1.0
            filename = os.path.basename(df['filename'].values[i])[:-4]
            if filename not in img_file_name_to_id:
                continue

            annotation = {
                "id": annotation_id,
                "bbox": gtboxes[i].tolist(),
                "image_id": img_file_name_to_id[filename],
                "segmentation": [],
                "ignore": 0,
                "visibility": visibility,
                "area": areas[i],
                "iscrowd": 0,
                "category_id": annotations['categories'][0]['id'],
                "seq": filename.split('_layer')[0],
                "track_id": int(track_ids[i]),
            }

            annotation_id += 1
            annotations['annotations'].append(annotation)

    # add "sequences" and "frame_range"
    annotations['sequences'] = seqs_names

    frame_range = {'start': 0.0, 'end': 1.0}
    annotations['frame_range'] = frame_range

    with open(annotation_file, 'w') as anno_file:
        json.dump(annotations, anno_file, indent=4)


if __name__ == '__main__':
    # generate_coco_from_spine(split_name='train_val', split='train_val')
    for split in ['train', 'val', 'test']:
        generate_coco_from_spine(split_name=split, split=split)

    # coco_dir = os.path.join('data/CrowdHuman', 'train_val')
    # annotation_file = os.path.join('data/CrowdHuman/annotations', 'train_val.json')
    # check_coco_from_mot(coco_dir, annotation_file, img_id=9012)
