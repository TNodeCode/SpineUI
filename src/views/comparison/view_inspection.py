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


def view_dataset_inspection():
    st.text("In this view you can inspect the annotations of your dataset. If you have multiple annotatioons for your dataset you can compare them here.")

    dataset_names = DatasetConfiguration.get_dataset_names()

    dataset_name = st.selectbox(
        "Select dataset",
        dataset_names,
    )

    if not dataset_name:
        return
    
    annotation_names = DatasetConfiguration.get_dataset_annotations(dataset_name)

    col1, col2 = st.columns(2)
    with col1:
        annotation_name_1 = st.selectbox(
            'Select first annotation',
            list(map(lambda x: x['name'], annotation_names))
        )
        show_bboxes_1 = st.checkbox(
            "Show Bounding Boxes Left",
            True
        )
        show_masks_1 = st.checkbox(
            "Show Segmentation Masks Left",
            True
        )
        instance_labels_1 = st.checkbox(
            "Show Instance IDs Left",
            False
        )
    with col2:
        annotation_name_2 = st.selectbox(
            'Select second annotation',
            list(map(lambda x: x['name'], annotation_names))
        )
        show_bboxes_2 = st.checkbox(
            "Show Bounding Boxes Right",
            True
        )
        show_masks_2 = st.checkbox(
            "Show Segmentation Masks Right",
            True
        )
        instance_labels_2 = st.checkbox(
            "Show Instance IDs Right",
            False
        )

    annotation_obj_1 = DatasetConfiguration.get_dataset_annotation(dataset_name=dataset_name, annotation_name=annotation_name_1)
    annotation_obj_2 = DatasetConfiguration.get_dataset_annotation(dataset_name=dataset_name, annotation_name=annotation_name_2)

    # Get all image paths that belong to a dataset
    image_paths = DatasetConfiguration.get_dataset_image_paths(dataset_name=dataset_name)

    # Create pagination object
    df = pd.DataFrame({"filename": image_paths})
    pagination = Pagination(df=df)

    # Create a selectbox for filename selection
    selected_filename = st.selectbox(
        'Select a filename',
        df['filename'].unique(),
        index=pagination.selected_index,
    )

    # Create detections objects
    detections_1 = DatasetAnnotations.get_detections(
        annotation_obj=annotation_obj_1,
        dataset_name=dataset_name,
        filename=selected_filename,
        instance_labels=instance_labels_1
    )
    detections_2 = DatasetAnnotations.get_detections(
        annotation_obj=annotation_obj_2,
        dataset_name=dataset_name,
        filename=selected_filename,
        instance_labels=instance_labels_2
    )

    pagination.update_selected_index(selected_filename)

    # Create buttons for selecting previous and next images
    col1, col2 = st.columns(2)
    col1.button('Previous', key='previous', on_click=pagination.backward)
    col2.button('Next', key='next', on_click=pagination.forward)

    # Load the image corresponding to the selected filename
    image_1 = Image.open(f"{selected_filename}").resize((512, 512))
    annotated_image_1 = Detection.plot_detections(image=image_1, detections=detections_1, show_bboxes=show_bboxes_1, show_masks=show_masks_1)
    image_2 = image_1.copy()
    annotated_image_2 = Detection.plot_detections(image=image_2, detections=detections_2, show_bboxes=show_bboxes_2, show_masks=show_masks_2)

    # Display the image with bounding boxes
    col1, col2 = st.columns(2)
    with col1:
        st.text(annotation_name_1)
        st.image(annotated_image_1)
    with col2:
        st.text(annotation_name_2)
        st.image(annotated_image_2)