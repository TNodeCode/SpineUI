import streamlit as st
import numpy as np
import pandas as pd
import globox
from os.path import basename
from PIL import Image, ImageDraw
from src.config.datasetconfig import DatasetConfiguration
from src.config.dataset_annotations import DatasetAnnotations
from src.datasets.masks import Masks
from src.util.detection import Detection
from src.views.pagination import Pagination


def view_tracking_results(dataset_name: str, traces_df: pd.DataFrame):
    st.text("In this view you can inspect the annotations of your dataset. If you have multiple annotatioons for your dataset you can compare them here.")

    if not dataset_name:
        return
    
    col1, col2 = st.columns(2)

    # Create pagination object
    pagination = Pagination(df=traces_df)

    # Create a selectbox for filename selection
    selected_filename = st.selectbox(
        'Select a filename',
        traces_df['filename'].unique(),
        index=pagination.selected_index,
    )

    image_traces = traces_df.query(f"filename == '{selected_filename}'")
    image_traces['x0'] = (image_traces['cx'] - 0.5 * image_traces['w']).astype(int)
    image_traces['x1'] = (image_traces['cx'] + 0.5 * image_traces['w']).astype(int)
    image_traces['y0'] = (image_traces['cy'] - 0.5 * image_traces['h']).astype(int)
    image_traces['y1'] = (image_traces['cy'] + 0.5 * image_traces['h']).astype(int)
    print("Image IDS", image_traces['object_id'].to_numpy())
    print("BBoxes", image_traces[['x0','y0','x1','y1']].to_numpy())

    # Create detections objects
    detections_1 = Detection.from_bboxes(bboxes=np.array([[0,0,0,0]]), class_id=np.array([0]))
    detections_2 = Detection.from_bboxes(bboxes=image_traces[['x0','y0','x1','y1']].to_numpy(), class_id=image_traces['object_id'].to_numpy())

    pagination.update_selected_index(selected_filename)

    # Create buttons for selecting previous and next images
    col1, col2 = st.columns(2)
    col1.button('Previous', key='previous', on_click=pagination.backward)
    col2.button('Next', key='next', on_click=pagination.forward)

    # Load the image corresponding to the selected filename
    image_1 = Image.open(f"{selected_filename}").resize((512, 512))
    annotated_image_1 = Detection.plot_detections(image=image_1, detections=detections_1, show_bboxes=True, show_masks=False)
    image_2 = image_1.copy()
    annotated_image_2 = Detection.plot_detections(image=image_2, detections=detections_2, show_bboxes=True, show_masks=False)

    # Display the image with bounding boxes
    col1, col2 = st.columns(2)
    with col1:
        st.text("FRAME T1")
        st.image(annotated_image_1)
    with col2:
        st.text("FRAME T2")
        st.image(annotated_image_2)