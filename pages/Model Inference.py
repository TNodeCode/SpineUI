import streamlit as st
import requests
import pandas as pd
from src.draw.bboxes import BBoxDrawer
from src.config.modelconfig import ModelConfiguration
from src.endpoints.endpoint import ModelEndpoint
from src.draw.image_loader import ImageLoader


def send_image(url: str, image_file):
    # Create the HTTP request
    url = f"{url}/image_inference"
    files = {"file": image_file}

    with st.spinner("Performing inference ..."):
        response = requests.post(url, files=files)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()
        return json_data["bboxes"]
    else:
        st.error("Error occurred during the request.")
        print(response.text)


def view_image_upload():
    st.markdown("# Model Inference")
    st.text("On this site you can run images through models")
    
    # Get configuration
    endpoints = ModelConfiguration.get_available_endpoints()

    selected_endpoint_name = st.selectbox(
        "Select Endpoint",
        list(map(lambda x: x['name'], endpoints)),
    )
    endpoint = list(filter(lambda x: x['name'] == selected_endpoint_name, endpoints))[0]

    available_models = ModelEndpoint.get_available_models(endpoint['url'])
    selected_model = st.selectbox("Select model", available_models)

    if selected_model:
        # Create an upload field
        image_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

        # Check if an image file was uploaded
        if image_file is not None:
            # Call the function to send the image
            bboxes = send_image(endpoint['url'], image_file)

            # Select a confidence score range
            min_score = st.slider("Minimum confidence score", min_value=0.0, max_value=1.0, step=0.01, value=0.5)

            # Render bounding boxes into image
            image_bboxes = BBoxDrawer.draw_bboxes(
                image=ImageLoader.load_from_uploaded_file(image_file),
                bboxes=bboxes,
                min_score=min_score
            )
            
            # Display image with bounding boxes
            st.image(image_bboxes, caption="Detected objects", use_column_width=True)

            # Display a table cotaining the bounding boxes
            df = pd.DataFrame(bboxes)
            st.dataframe(df)

view_image_upload()