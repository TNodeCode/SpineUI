from ultralytics import YOLO
import pandas as pd
import glob
import os

def load_model(dataset_name, model_name):
    model = YOLO(f"models/{dataset_name}_{model_name}.pt")
    return model

def predict(dataset_name, subset, model, model_name):
    # Run batched inference on a list of images
    filenames = glob.glob(f"datasets/{dataset_name}/{subset}/images/*.png")
    results = model(filenames)  # return a list of Results objects

    df_data = []

    # Process results list
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy.to(int)  # Boxes object for bbox outputs
        labels = result.boxes.cls.to(int)  # Box labels
        scores = result.boxes.conf  # Box scores
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        for j in range(boxes.shape[0]):
            data = {
                "filename": os.path.basename(filenames[i]),
                "xmin": int(boxes[j][0]),
                "xmax": int(boxes[j][2]),
                "ymin": int(boxes[j][1]),
                "ymax": int(boxes[j][3]),
                "class_name": int(labels[j]),
                "class_index": int(labels[j]),
                "score": float(scores[j])
            }
            df_data.append(data)

    return df_data
