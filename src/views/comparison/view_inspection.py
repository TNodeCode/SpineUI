import streamlit as st
import numpy as np
import pandas as pd
import globox
from os.path import basename
from PIL import Image, ImageDraw
from src.config.datasetconfig import DatasetConfiguration
from src.datasets.masks import Masks
from src.util.detection import Detection
from src.views.pagination import Pagination


def get_mask_detections(dataset_name: str, annotation_name: str, selected_filename: str):
    # Get all mask image paths that belong to a dataset
    mask_images = DatasetConfiguration.get_dataset_mask_image_paths(
        dataset_name=dataset_name,
        annotation_name=annotation_name
    )
    # Get mask image paths
    mask_image_paths = list(filter(lambda x: basename(selected_filename) in x, mask_images))
    # Create detection objects
    detections = Masks.mask_images_to_detection(mask_image_paths)
    return detections


def get_coco_detections(annotation_file: str, selected_filename: str):
    coco = globox.AnnotationSet.from_coco(file_path=annotation_file)
    annotations = list(coco)
    annotation = list(filter(lambda x: basename(x.image_id) == basename(selected_filename), annotations))[0]
    bboxes = np.array(list(map(lambda b: [b.xmin, b.ymin, b.xmax, b.ymax], annotation.boxes)))
    return Detection.from_bboxes(bboxes)


def get_detections(annotation_obj: object, dataset_name: str, selected_filename: str):
    if annotation_obj['type'] == 'coco':
        return get_coco_detections(annotation_file=annotation_obj['paths'][0], selected_filename=selected_filename)
    elif annotation_obj['type'] == 'masks':
        return get_mask_detections(dataset_name=dataset_name, annotation_name=annotation_obj['name'], selected_filename=selected_filename)


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
    detections_1 = get_detections(annotation_obj=annotation_obj_1, dataset_name=dataset_name, selected_filename=selected_filename)
    detections_2 = get_detections(annotation_obj=annotation_obj_2, dataset_name=dataset_name, selected_filename=selected_filename)

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