import os
import streamlit as st
import requests
import pandas as pd
from src.draw.bboxes import BBoxDrawer
from src.config.datasetconfig import DatasetConfiguration
from src.config.modelconfig import ModelConfiguration
from src.endpoints.endpoint import ModelEndpoint
from src.draw.image_loader import ImageLoader


def send_image(url: str, image_file, model_id: str):
    # Create the HTTP request
    url = f"{url}/image_inference/{model_id}"
    files = {"file": image_file}

    with st.spinner("Performing inference ..."):
        response = requests.post(url, data={"model_id": model_id}, files=files)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()
        return json_data["bboxes"]
    else:
        st.error("Error occurred during the request.")


def get_detections_filename(model_name: str, dataset_name: str):
    return f"detections/{model_name}_{dataset_name}.csv"


def get_filenames_already_predicted(model_name: str, dataset_name: str) -> list[str]:
    filepath = get_detections_filename(model_name=model_name, dataset_name=dataset_name)
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        return list(df['filename'].unique())
    else:
        return []


def update_detections(model_name: str, dataset_name: str, new_df: pd.DataFrame):
    detections_filename = get_detections_filename(model_name=model_name, dataset_name=dataset_name)
    df = None
    if os.path.exists(detections_filename):
        df = pd.read_csv(detections_filename)
        df = pd.concat([df, new_df])
    else:
        df = new_df
    df = df.drop_duplicates()
    df.to_csv(detections_filename, index=False)


def view_dataset_upload():
    st.markdown("# Dataset Inference")
    st.text("On this site you can run a complete dataset through a model")
    
    # Get configuration
    datasets = DatasetConfiguration.get_dataset_names()
    endpoints = ModelConfiguration.get_available_endpoints()

    selected_endpoint_name = st.selectbox(
        "Select Endpoint",
        list(map(lambda x: x['name'], endpoints)),
    )
    endpoint = list(filter(lambda x: x['name'] == selected_endpoint_name, endpoints))[0]

    available_models = ModelEndpoint.get_available_models(endpoint['url'])
    selected_model = st.selectbox("Select model", available_models)
    selected_dataset = st.selectbox("Select dataset", datasets)

    # Select a confidence score range
    min_score = st.slider("Minimum confidence score", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

    progress_text = st.empty()
    progress_bar = st.empty()
    displayed_image = st.empty()
    displayed_df = st.empty()

    if selected_dataset and selected_model:
        paths = DatasetConfiguration.get_dataset_image_paths(selected_dataset)
        already_predicted_files = get_filenames_already_predicted(model_name=selected_model, dataset_name=selected_dataset)
        paths = list(filter(lambda x: os.path.basename(x) not in already_predicted_files, paths))
        inference_button = st.button("Perform Inference on Dataset")
        
        # Check if an image file was uploaded
        if inference_button:
            progress_bar.progress(0)
            for i, path in enumerate(paths):
                progress_text.text(f"Image {i+1} / {len(paths)}")
                with open(path, "rb") as image_file:
                    # Call the function to send the image
                    bboxes = send_image(url=endpoint['url'], image_file=image_file, model_id=selected_model)
                    df_boxes_new = pd.DataFrame(bboxes)
                    update_detections(model_name=selected_model, dataset_name=selected_dataset, new_df=df_boxes_new)

                    # Render bounding boxes into image
                    image_bboxes = BBoxDrawer.draw_bboxes(
                        image=ImageLoader.load_from_uploaded_file(image_file),
                        bboxes=bboxes,
                        min_score=min_score
                    )
                    
                    # Display image with bounding boxes
                    displayed_image.image(image_bboxes, caption="Detected objects", use_column_width=True)

                    # Display a table cotaining the bounding boxes
                    df = pd.DataFrame(bboxes)
                    displayed_df.dataframe(df)
                    progress_bar.progress((i+1) / len(paths))


view_dataset_upload()