import os
import time
import torch
import yaml
import zipfile
import mmengine
import mmcv
import numpy as np
from mmdet.apis import inference_detector, init_detector, init_track_model, inference_mot
from fastapi import FastAPI, UploadFile, File
from mmdet.structures.track_data_sample import TrackDataSample



device = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE", device)

available_models = {}
current_model_id = ""
current_model = None
current_model_id = ""
current_reid = None

def get_available_models():
    global available_models
    #if (len(available_models.keys())) > 0:
    #    return
    # read available models
    with open('available_models.yml', 'r') as file:
        available_models = yaml.safe_load(file)
        return available_models

def update_model(model_id: str):
    with open('available_models.yml', 'r') as file:
        available_models = yaml.safe_load(file)
    global current_model_id
    global model
    if current_model_id != model_id:
        print("UPDATING MODEL ...")
        current_model_id = model_id
        current_model = list(filter(lambda x: x['name'] == current_model_id, available_models))[0]
        print("CURRENT MODEL", current_model)
        # Reload model
        model = init_detector(
            current_model["config"],
            current_model["weights"],
            device=device
        )
    return model

def update_mot(model_id: str):
    mmengine.registry.init_default_scope('mmdet')
    with open('available_models.yml', 'r') as file:
        available_models = yaml.safe_load(file)
    global current_model_id
    global reid
    if current_model_id != model_id:
        print("UPDATING MODEL ...")
        current_model_id = model_id
        current_reid = list(filter(lambda x: x['name'] == current_model_id, available_models))[0]
        print("CURRENT REID", current_reid)
        # Reload model
        reid = init_track_model(
            current_reid["config"],
            None,
            detector=current_reid['detector_weights'],
            reid=current_reid['reid_weights'],
            device=device,
            cfg_options={},
        )
    return reid

app = FastAPI()


@app.get("/available_models")
def get_available_models():
    with open('available_models.yml', 'r') as file:
        available_models = yaml.safe_load(file)
    return available_models


def predict_images(filenames):
    global model
    results = inference_detector(
        model=model,
        imgs=filenames
    )

    detections = []
    for filename, result in zip(filenames, results):
        print(f"Processing file {filename} ...")
        bboxes = result.pred_instances.bboxes
        labels = result.pred_instances.labels
        scores = result.pred_instances.scores
        # Iterate over detected bounding boxes
        for ((x0, y0, x1, y1), label, score) in zip(bboxes, labels, scores):
            if score >= 0.5:
                detections.append({
                    "filename": filename,
                    "class_index": 0,
                    "class_name": "spine",
                    "xmin": int(x0),
                    "ymin": int(y0),
                    "xmax": int(x1),
                    "ymax": int(y1),
                    "score": float(score)
                })
    return detections


def predict_tracks(filenames):
    global reid
    detections, trajectories = [], []
    for i, filename in enumerate(filenames):
        img = mmcv.imread(filename)
        result: TrackDataSample = inference_mot(
            reid,
            img,
            frame_id=i,
            video_len=len(filenames)
        )
        if result.video_data_samples[0].pred_instances.bboxes.shape[0] > 0: # check if there are any detections
            # Detection results
            if result.video_data_samples[0].pred_instances.bboxes.shape[1] == 4:
                bboxes = result.video_data_samples[0].pred_instances.bboxes
                labels = result.video_data_samples[0].pred_instances.labels
                scores = result.video_data_samples[0].pred_instances.scores          
            else:
                raise Exception("Invalid bboxes shape")
                
            for j, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
                if bool(bbox.min().isinf()) or bool(bbox.max().isinf() or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                    continue
                if type(bbox) == np.ndarray and (bool(np.isinf(bbox.min())) or bool(np.isinf(bbox.max())) or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                    continue
                detections.append({
                    'filename': os.path.basename(filename),
                    'frame': i+1,
                    'x0': int(bbox[0]),
                    'x1': int(bbox[2]),
                    'y0': int(bbox[1]),
                    'y1': int(bbox[3]),
                    'score': float(score)
                })
            
        if result.video_data_samples[0].pred_track_instances.bboxes.shape[0] > 0: # check if there are any trajectories
            if result.video_data_samples[0].pred_track_instances.bboxes.shape[1] == 4:
                bboxes = result.video_data_samples[0].pred_track_instances.bboxes
                labels = result.video_data_samples[0].pred_track_instances.labels
                instances_ids = result.video_data_samples[0].pred_track_instances.instances_id
                scores = result.video_data_samples[0].pred_track_instances.scores                
            elif result.video_data_samples[0].pred_track_instances.bboxes.shape[1] == 7:
                bboxes = result.video_data_samples[0].pred_track_instances.bboxes[:, 2:6]
                # with to x1 = x0 + width
                bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
                # height to y1 = y0 + height
                bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
                labels = result.video_data_samples[0].pred_track_instances.bboxes[:, 0]
                instances_ids = result.video_data_samples[0].pred_track_instances.bboxes[:, 1]
                scores = result.video_data_samples[0].pred_track_instances.bboxes[:, 6]
            else:
                raise Exception("Invalid bboxes shape")
                
            for j, (bbox, label, instance_id, score) in enumerate(zip(bboxes, labels, instances_ids, scores)):
                if type(bbox) == torch.Tensor and (bool(bbox.min().isinf()) or bool(bbox.max().isinf()) or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                    continue
                if type(bbox) == np.ndarray and (bool(np.isinf(bbox.min())) or bool(np.isinf(bbox.max())) or int(bbox.min()) < 0 or int(bbox.max()) > img.shape[0]):
                    continue
                trajectories.append({
                    'filename': os.path.basename(filename),
                    'frame': i+1,
                    'object_id': int(instance_id),                        'x0': int(bbox[0]),
                    'x1': int(bbox[1]),
                    'w': int(bbox[2]) - int(bbox[0]),
                    'h': int(bbox[3]) - int(bbox[1]),
                    'score': float(score)
                })
        
    return detections, trajectories

@app.post("/image_inference/{model_id}")
async def image_inference(model_id: str, file: UploadFile = File(...)):
    model = update_model(model_id=model_id)
    
    # Specify the directory where you want to save the image
    save_directory = "data/tmp"

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save the uploaded image to the specified directory
    file_path = os.path.join(save_directory, file.filename)
    
    with open(file_path, "wb") as image_file:
        content = await file.read()
        image_file.write(content)

    print("Compute tracks ...")
    time_start = time.time()
    predictions = predict_images([file_path])
    time_end = time.time()
    print("Inference took " + str(time_end-time_start) + " seconds")
    os.remove(file_path)

    return {
        "filename": file.filename,
        "saved_path": file_path,
        "bboxes": predictions,
    }

@app.post("/tracking_inference/{model_id}")
async def tracking_inference(model_id: str, file: UploadFile = File(...)):
    model = update_mot(model_id=model_id)

    # Specify the directory where you want to save the image
    save_directory = "data/tmp"

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save the uploaded image to the specified directory
    file_path = os.path.join(save_directory, file.filename)
    with open(file_path, "wb") as zip_file:
        content = await file.read()
        zip_file.write(content)

    with zipfile.ZipFile(file_path,"r") as zip_ref:
        zip_ref.extractall(save_directory)

    # delete ZIP file
    os.remove(file_path)

    print("Inference ...")
    time_start = time.time()
    image_files = list(map(lambda x: save_directory + "/" + x, os.listdir(save_directory)))
    _, trajectories = predict_tracks(image_files)
    time_end = time.time()
    print("Inference took " + str(time_end-time_start) + " seconds")

    # remove images
    for image_file in image_files:
        os.remove(image_file)

    return {
        "filename": file.filename,
        "saved_path": file_path,
        "trajectories": trajectories,
    }
