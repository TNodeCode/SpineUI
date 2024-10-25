import os
import shutil
import pandas as pd

# Here are the original images and annotations
images_dir = "data/csv/images" if not os.getenv("IMAGES_DIR") else os.getenv("IMAGES_DIR")
annotations_dir = "data/csv/annotations" if not os.getenv("ANNOTATIONS_DIR") else os.getenv("ANNOTATIONS_DIR")

# Target directory
output_dir = "data/csv/mot" if not os.getenv("OUTPUT_DIR") else os.getenv("OUTPUT_DIR")


filenames = os.listdir(annotations_dir)
for filename in filenames:
    print("Processing file", filename)
    # Read CSV annotations
    df_annotations = pd.read_csv(f"{annotations_dir}/{filename}")
    if df_annotations.shape[0] == 0:
        continue
    df_annotations['filename'] = df_annotations['filename'].apply(os.path.basename)
    image_files = df_annotations['filename'].unique()
    output_dir_mot = f"{output_dir}/{os.path.basename(filename)[:-4]}"

    # Make MOT directories
    os.makedirs(output_dir_mot, exist_ok=True)
    os.makedirs(f"{output_dir_mot}/det", exist_ok=True)
    os.makedirs(f"{output_dir_mot}/gt", exist_ok=True)
    os.makedirs(f"{output_dir_mot}/img", exist_ok=True)
    
    # Collect trajectory boxes
    bboxes = []
    for image_id, image_file in enumerate(image_files):
        df_bboxes = df_annotations[df_annotations['filename'] == image_file]
        for j, bbox in df_bboxes.iterrows():
            bboxes.append({
                "frame": int(image_id) + 1,
                "track_id": int(bbox['id']),
                "xmin": int(bbox['xmin']),
                "ymin": int(bbox['ymin']),
                "width": int(bbox['xmax']) - int(bbox['xmin']),
                "height": int(bbox['ymax']) - int(bbox['ymin']),
                "conf": 1,
                "class_id": 0,
                "visibility": 1,
            })

        src = f"{images_dir}/{os.path.basename(image_file)}"
        if not os.path.exists(src):
            raise Exception(f"{src} does not exist")
        
        # Copy image file to target directory
        shutil.copy2(
            src=src,
            dst=f"{output_dir_mot}/img/{str(image_id+1).zfill(6)}.png"
        )

        # Create seqinfo
        seqinfo = f"""[Sequence]
name={os.path.basename(filename[:-4])}
imDir=img
frameRate=1
seqLen={len(image_files)}
imWidth=512
imHeight=512
imExt=.png
seqLength={len(image_files)}
"""
        with open(f"{output_dir_mot}/seqinfo.ini", "w+") as f:
            f.write(seqinfo)

    # Create gt.txt
    df_mot = pd.DataFrame(bboxes)
    df_mot.to_csv(f"{output_dir_mot}/gt/gt.txt", index=False, header=False)

    # Create det.txt
    df_mot['track_id'] = -1
    df_mot.to_csv(f"{output_dir_mot}/det/det.txt", index=False, header=False)