import os
import glob
import time
import numpy as np
import pandas as pd
from mmdet.apis import inference_detector, init_detector


def detect(
        model_type: str,
        model_name: str,
        weight_file: str,
        results_file: str,
        dataset_dir: str,
        image_files: str,
        batch_size: int = 8,
        score_threshold: float = 0.5,
        device: str = "cuda:0",
        work_dir: str = "./work_dirs"
):
    config_file_path = f"{work_dir}/{model_name}.py"
    weight_file_path = f"{work_dir}/{weight_file}"

    print("Config file", config_file_path)
    print("Weight file", weight_file_path)
    print("Device", device)
    print("Batch Size", batch_size)

    # Initialize the model
    print("Loading model ...")
    model = init_detector(
        config_file_path,
        weight_file_path,
        device=device
    )
    print("Model loaded")

    filenames = glob.glob(os.path.join(dataset_dir, image_files))
    n_files = len(filenames)
    n_batches=(n_files // batch_size) + 1

    detected_bboxes = []
    durations = []
    for b in range((n_files // batch_size) + 1):
        if (len(filenames[b*batch_size:(b+1)*batch_size])) < 1:
            continue
        # Send images through model
        print(f"Processing batch {b}/{n_batches} ...")
        start = time.time()
        results = inference_detector(
            model=model,
            imgs=filenames[b*batch_size:(b+1)*batch_size]
        )
        end = time.time()
        durations.append(end - start)
        print("Finished batch in", end - start, "seconds")
        # Iterate over image results
        for i, result in enumerate(results):
            filename = filenames[b*batch_size+i].replace(os.sep, '/').replace(dataset_dir, "")
            bboxes = result.pred_instances.bboxes
            labels = result.pred_instances.labels
            scores = result.pred_instances.scores
            # Iterate over detected bounding boxes
            for ((x0, y0, x1, y1), label, score) in zip(bboxes, labels, scores):
                if score >= score_threshold:
                    detected_bboxes.append({
                        "filename": filename,
                        "class_index": int(label),
                        "class_name": "spine",
                        "xmin": int(x0),
                        "ymin": int(y0),
                        "xmax": int(x1),
                        "ymax": int(y1),
                        "score": float(score)
                    })

    # Create the CSV file that contains the detections
    csv_filename = f"{work_dir}/{results_file}"
    os.makedirs(os.path.split(csv_filename)[0], exist_ok=True)
    pd.DataFrame(detected_bboxes).to_csv(csv_filename, index=False)
    print("Saved CSV file at", csv_filename)

    # Print some statistics about the detection process
    durations = np.array(durations)
    print("Inference took", durations.mean(), "per batch on average, std=", durations.std())
    print("Inference took", durations.mean() / batch_size, "per image on average, std=", durations.std() / batch_size)