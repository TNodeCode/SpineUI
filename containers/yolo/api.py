import os
import json
import time
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from predict_boxes import load_model, predict

# Load a model
dataset_name = "spine"
model_name = "yolov8x"
subset = "tmp"

app = FastAPI()
model = load_model(dataset_name=dataset_name, model_name=model_name)


@app.get("/available_models")
async def get_available_models():
    return [
        "yolov8x",
        "yolov9e",
        "gelan-e",
    ]


@app.post("/image_inference/")
async def upload_image(file: UploadFile = File(...)):
    time_start = time.time()
    print("START", time_start)

    # Specify the directory where you want to save the image
    save_directory = "datasets/spine/tmp/images"

    # Create the directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Save the uploaded image to the specified directory
    file_path = os.path.join(save_directory, file.filename)
    with open(file_path, "wb") as image_file:
        content = await file.read()
        image_file.write(content)

    df_data = predict(dataset_name=dataset_name, subset=subset, model=model, model_name=model_name)
    #df = pd.DataFrame(df_data)
    #df.to_csv(f"{dataset_name}_{model_name}_{subset}.csv", index=False)
    #df.to_json()

    os.remove(file_path)

    return {
        "filename": file.filename,
        "saved_path": file_path,
        "bboxes": df_data
    }