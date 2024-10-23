import os
import streamlit as st
import requests
import pandas as pd
import zipfile
import numpy as np
from PIL import Image
from src.config.dataset_annotations import DatasetAnnotations
from src.config.modelconfig import ModelConfiguration
from src.endpoints.endpoint import ModelEndpoint
from src.draw.image_loader import ImageLoader
from src.views.pagination import Pagination
from src.util.detection import Detection


def send_stack(url: str, zip_file: str, model_name: str):
    # Create the HTTP request
    url = f"{url}/tracking_inference/{model_name}"
    files = {"file": zip_file}

    with st.spinner("Performing inference ..."):
        response = requests.post(url, files=files)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()
        return json_data["trajectories"]
    else:
        st.error("Error occurred during the request.")
        print(response.text)


def save_tracking_inference(df: pd.DataFrame, results_file_path: str):
    stack_name = results_file_path.split("/")[-1]
    n_images = df['frame'].unique().shape[0]
    det_file_content = []
    for i, row in df.iterrows():
        det_file_content.append(f"{row['frame']},{row['object_id']},{row['x0']},{row['x1']},{row['w']},{row['h']},-1,-1,-1")
    det_file_content = "\n".join(det_file_content)
    os.makedirs(results_file_path + "/det", exist_ok=True)
    os.makedirs(results_file_path + "/gt", exist_ok=True)
    with open(f"{results_file_path}/det/det.txt", "w+") as fp:
        fp.write(det_file_content)
    with open(f"{results_file_path}/gt/gt.txt", "w+") as fp:
        fp.write(det_file_content)
    seqinfo_content = f"[Sequence]\nname={stack_name}\nframeRate=1\nseqLen={n_images}\nimWidth=512\nimHeight=512\nimExt=.png\nseqLength={n_images}"
    with open(results_file_path + "/seqinfo.ini", "w+") as fp:
        fp.write(seqinfo_content)


def load_trajectories_from_detections(save_path: str):
    with open(f"{save_path}/det/det.txt", "r") as fp:
        content = fp.read()
    image_files = os.listdir(f"{save_path}/img")
    lines = content.split("\n")
    trajectories = []
    for i, line in enumerate(lines):
        data = line.split(",")
        trajectories.append({
            'filename': image_files[int(data[0])-1],
            'frame': int(data[0]),
            'object_id': int(data[1]),
            'x0': int(data[2]),
            'x1': int(data[3]),
            'w': int(data[4]),
            'h': int(data[5]),
        })
    return pd.DataFrame(trajectories)



def view_stack_upload():
    st.markdown("# Model Inference")
    st.text("On this site you can run images through tracking models")
    
    # Get configuration
    endpoints = ModelConfiguration.get_available_endpoints()

    selected_endpoint_name = st.selectbox(
        "Select Endpoint",
        list(map(lambda x: x['name'], endpoints)),
    )
    endpoint = list(filter(lambda x: x['name'] == selected_endpoint_name, endpoints))[0]

    available_models = ModelEndpoint.get_available_models(endpoint['url'])
    available_models = list(filter(lambda x: x['type'] == 'tracking', available_models))
    available_models = list(map(lambda x: x['name'], available_models))
    selected_model = st.selectbox("Select model", available_models)

    if selected_model:
        btn_save_results = st.checkbox("Save results")
        save_path = st.text_input(label="Save dir", value=f"./datasets/detections/{selected_model}/stack_name")

        if not os.path.exists(save_path):
            # Create an upload field
            zip_file = st.file_uploader("Upload image stack", type=["zip"])

        # Check if an image file was uploaded
        if os.path.exists(save_path) or zip_file is not None:
            if not os.path.exists(save_path):
                # Call the function to send the image
                trajectories = send_stack(
                    url=endpoint['url'],
                    zip_file=zip_file,
                    model_name=selected_model
                )
            else:
                trajectories = load_trajectories_from_detections(save_path)

            # Display a table cotaining the bounding boxes
            df = pd.DataFrame(trajectories)
            st.dataframe(df)

            if btn_save_results:
                img_path = save_path + "/img"
                if not os.path.exists(save_path):
                    save_tracking_inference(df=df, results_file_path=save_path)
                    os.makedirs(img_path, exist_ok=True)
                    zipfile.ZipFile(zip_file).extractall(img_path)

                # Create pagination object
                df_filenames = pd.DataFrame({"filename": df['filename']})
                print("DF", df)
                print("DF FILENAMES", df_filenames)
                pagination = Pagination(df=df)

                # Create a selectbox for filename selection
                selected_filename = st.selectbox(
                    'Select a filename',
                    df_filenames['filename'].unique(),
                    index=pagination.selected_index,
                )

                pagination.update_selected_index(os.path.basename(selected_filename))

                # Create buttons for selecting previous and next images
                col1, col2 = st.columns(2)
                col1.button('Previous', key='previous', on_click=pagination.backward)
                col2.button('Next', key='next', on_click=pagination.forward)

                bboxes = []
                class_ids = []
                for i, row in df[df['filename'] == os.path.basename(selected_filename)].iterrows():
                    bboxes.append([row['x0'], row['x1'], row['x0']+row['w'], row['x1']+row['h']])
                    class_ids.append(row['object_id'])
                detections_1 = Detection.from_bboxes(np.array(bboxes), class_id=np.array(class_ids))

                # Load the image corresponding to the selected filename
                image_1 = Image.open(f"{img_path}/{selected_filename}").resize((512, 512))
                annotated_image_1 = Detection.plot_detections(
                    image=image_1,
                    detections=detections_1,
                    show_bboxes=True,
                    show_masks=True,
                    show_labels=True,
                )
                st.image(annotated_image_1)

view_stack_upload()