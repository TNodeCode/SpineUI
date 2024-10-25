import pandas as pd
from PIL import Image
import shutil
import json
import os



root_dir = 'data/MOT17'
out_dir = 'data/crowdhuman'
det_db_file = f"data/det_db_spine.json"
detections_dict = {}

os.makedirs(out_dir, exist_ok=True)
os.makedirs(f"{out_dir}/Images", exist_ok=True)

for split in ['train', 'val', 'test']:
    detections_txt = []
    stacks = os.listdir(f"{root_dir}/{split}")
    for stack in stacks:
        box_id_occurences = {}
        stack_dir = f"{root_dir}/{split}/{stack}"
        if not os.path.isdir(stack_dir):
            continue
        df = pd.read_csv(f"{stack_dir}/gt/gt.txt", header=None)
        filenames = df[0].unique()
        image_names = os.listdir(f"{stack_dir}/img")
        for filename in filenames:
            image_name = image_names[filename - 1]
            df_img = df[df[0] == filename]
            image_id, image_ext = os.path.splitext(os.path.basename(image_name))
            image_id = f"{stack}_{image_id}"
            img_obj = {
                'ID': image_id,
                'gtboxes': []
            }
            for i, bbox in df_img.iterrows():
                box_id = int(bbox[1])
                if box_id in box_id_occurences.keys():
                    box_id_occurences[box_id] = box_id_occurences[box_id] + 1
                else:
                    box_id_occurences.update({box_id: 0})
                coords = [int(bbox[2]), int(bbox[3]), int(bbox[4]), int(bbox[5])]
                bbox_obj = {
                    'tag': 'person',
                    'vbox': coords,
                    'fbox': coords,
                    'hbox': coords,
                    'extra': {
                        'box_id': box_id,
                        'occ': box_id_occurences[box_id],
                    },
                    'head_attr': {
                        'ignore': 1,
                        'occ': 0,
                        'unsure': 0
                    }
                }
                img_obj['gtboxes'].append(bbox_obj)  
                # Create bbox annotations fro det_db.json fie
                bboxes_str_list = []
                for i, bbox in df_img.iterrows():
                    bbox_str = f"{float(bbox[2])},{float(bbox[3])},{float(bbox[4])},{float(bbox[5])},1.0"
                    bboxes_str_list.append(bbox_str)
                detections_dict.update({f"crowdhuman/train_image/{image_id}.txt": bboxes_str_list})
                detections_dict.update({f"DanceTrack/{split}/{stack}/img1/{str(filename).zfill(8)}.txt": bboxes_str_list})
                detections_txt.append(json.dumps(img_obj))
            Image.open(f"{stack_dir}/img/{image_names[filename - 1]}") \
                .convert('RGB') \
                .save(f"{out_dir}/Images/{image_id}.jpg")

    with open(f"{out_dir}/annotation_{split}.odgt", 'w+') as f:
        f.write("\n".join(detections_txt) + "\n")

# Create annotation files
filenames = [f"{out_dir}/annotation_train.odgt", f"{out_dir}/annotation_val.odgt"]
with open(f"{out_dir}/annotation_trainval.odgt", 'w+') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)

# Create det_db file
with open(det_db_file, 'w+') as f:
	f.write(json.dumps(detections_dict, indent=4))