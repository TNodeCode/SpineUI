import os
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from src.config.datasetconfig import DatasetConfiguration
from src.config.dataset_annotations import DatasetAnnotations

from src.views.pagination import Pagination


def view_model_comparison():
    # Select a dataset
    dataset_name = st.selectbox(
        "Select Dataset",
        DatasetConfiguration.get_dataset_names()
    )

    # Get dataset configuration
    dataset = DatasetConfiguration.get_dataset_config(dataset_name=dataset_name)
    # Get paths of all images belonging to the selected dataset
    image_paths = DatasetConfiguration.get_dataset_image_paths(dataset_name=dataset_name)
    # Contains mappings from image file names to the directories where they are stored
    image_path_mappings = {x.replace(os.sep, '/').replace(dataset['root_dir'], '')[1:] : x for x in image_paths}

    detection_objs = DatasetConfiguration.get_dataset_detections(dataset_name=dataset_name)
    
    # if there are no CSV files show a warning and return
    if len(detection_objs) < 1:
        st.warning('No detections available for this dataset', icon="⚠️")
        return

    detection_names = list(map(lambda x: x["name"], detection_objs))

    # Display columns for comparing detections
    csv_col1, csv_col2 = st.columns(2)
    with csv_col1:
        detection_name_1 = st.selectbox(
            "Select Detection Model 1",
            detection_names
        )
        # Get all available CSV files
        csv_files_1 = DatasetConfiguration.get_detection_csv_files(
            dataset_name=dataset_name,
            detection_name=detection_name_1
        )
        # if there are no CSV files show a warning and return
        if len(csv_files_1) < 1:
            st.warning('No CSV detection result files found', icon="⚠️")
        else:
            csv_filename_1 = st.selectbox(
                'Select first CSV file',
                csv_files_1
            )
            show_gt_bboxes_1 = st.checkbox(
                "Show Ground Truth Left",
                True
            )
            show_pred_bboxes_1 = st.checkbox(
                "Show Predictions Left",
                True
            )
    with csv_col2:
        detection_name_2 = st.selectbox(
            "Select Detection Model 2",
            detection_names
        )
        # Get all available CSV files
        csv_files_2 = DatasetConfiguration.get_detection_csv_files(
            dataset_name=dataset_name,
            detection_name=detection_name_2
        )
        # if there are no CSV files show a warning and return
        if len(csv_files_1) < 1:
            st.warning('No CSV detection result files found', icon="⚠️")
        else:
            csv_filename_2 = st.selectbox(
                'Select second CSV file',
                csv_files_2
            )
            show_gt_bboxes_2 = st.checkbox(
                "Show Ground Truth Right",
                True
            )
            show_pred_bboxes_2 = st.checkbox(
                "Show Predictions Right",
                True
            )

    # Load the DataFrame
    df_1 = pd.read_csv(csv_filename_1)
    if not "score" in df_1.columns:
        df_1["score"] = 1.0
    df_2 = pd.read_csv(csv_filename_2)
    if not "score" in df_2.columns:
        df_2["score"] = 1.0

    pagination = Pagination(df=df_1)

    # Create a selectbox for filename selection
    selected_filename = st.selectbox(
        'Select a filename',
        df_1['filename'].unique(),
        index=pagination.selected_index,
    )

    # Select a ground truth annotation
    annotations = DatasetConfiguration.get_dataset_annotations(dataset_name)
    annotation_name_1 = st.selectbox(
        'Select Ground Truth',
        list(map(lambda x: x['name'], annotations))
    )
    annotation = DatasetConfiguration.get_dataset_annotation(dataset_name=dataset_name, annotation_name=annotation_name_1)

    gt_annotations = DatasetAnnotations.get_detections(annotation_obj=annotation, dataset_name=dataset_name, filename=selected_filename)
    pagination.update_selected_index(selected_filename)

    # Create buttons for selecting previous and next images
    col1, col2 = st.columns(2)
    col1.button('Previous', key='previous', on_click=pagination.backward)
    col2.button('Next', key='next', on_click=pagination.forward)

    # Filter the DataFrame based on the selected filename
    filtered_df_1 = df_1[df_1['filename'] == selected_filename]
    filtered_df_2 = df_2[df_2['filename'] == selected_filename]

    # Load the image corresponding to the selected filename
    basepath = image_path_mappings[selected_filename]
    image_1 = Image.open(basepath).resize((512, 512))
    image_1 = image_1.convert(mode='RGB')
    image_2 = image_1.copy()

    # Create a new image with bounding boxes drawn
    draw_1 = ImageDraw.Draw(image_1)
    draw_2 = ImageDraw.Draw(image_2)

    # Define color mappings for different classes
    color_mappings = {
        'class1': '#00ff00',
        'class2': 'blue',
        'class3': 'green',
        # Add more class-color mappings as needed
    }

    # Draw ground truth bounding boxes and labels on the image
    if show_gt_bboxes_1:
        for xmin, ymin, xmax, ymax in gt_annotations.xyxy:
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            color = 'green'  # Default to yellow if class color not defined
            draw_1.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

    # Draw bounding boxes and labels on the image
    if show_pred_bboxes_1:
        for _, row in filtered_df_1.iterrows():
            if row['score'] < 0.5:
                continue
            class_name = row['class_name']
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            color = color_mappings.get(class_name, 'yellow')  # Default to yellow if class color not defined
            draw_1.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

    # Draw ground truth bounding boxes and labels on the image
    if show_gt_bboxes_2:
        for xmin, ymin, xmax, ymax in gt_annotations.xyxy:
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            color = 'green'  # Default to yellow if class color not defined
            draw_2.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

    # Draw bounding boxes and labels on the image
    if show_pred_bboxes_2:
        for _, row in filtered_df_2.iterrows():
            if row['score'] < 0.5:
                continue
            class_name = row['class_name']
            xmin = row['xmin']
            ymin = row['ymin']
            xmax = row['xmax']
            ymax = row['ymax']
            color = color_mappings.get(class_name, 'yellow')  # Default to yellow if class color not defined
            draw_2.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)

    # Display the image with bounding boxes
    col1, col2 = st.columns(2)
    with col1:
        st.text(csv_filename_1)
        st.image(image_1)
    with col2:
        st.text(csv_filename_2)
        st.image(image_2)