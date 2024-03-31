import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from src.views.pagination import Pagination


def view_model_comparison(basepath: str, cvs_files: list[str]):
    csv_col1, csv_col2 = st.columns(2)
    with csv_col1:
        csv_filename_1 = st.selectbox(
            'Select first CSV file',
            cvs_files
        )
    with csv_col2:
        csv_filename_2 = st.selectbox(
            'Select second CSV file',
            cvs_files
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

    pagination.update_selected_index(selected_filename)

    # Create buttons for selecting previous and next images
    col1, col2 = st.columns(2)
    col1.button('Previous', key='previous', on_click=pagination.backward)
    col2.button('Next', key='next', on_click=pagination.forward)

    # Filter the DataFrame based on the selected filename
    filtered_df_1 = df_1[df_1['filename'] == selected_filename]
    filtered_df_2 = df_2[df_2['filename'] == selected_filename]

    # Load the image corresponding to the selected filename
    image_1 = Image.open(f"{basepath}/{selected_filename}").resize((512, 512))
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

    # Draw bounding boxes and labels on the image
    for _, row in filtered_df_1.iterrows():
        if row['score'] < 0.5:
            continue
        class_name = row['class']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        color = color_mappings.get(class_name, 'yellow')  # Default to yellow if class color not defined
        draw_1.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
        #draw.text((xmin, ymin-15), class_name, fill=color)

    # Draw bounding boxes and labels on the image
    for _, row in filtered_df_2.iterrows():
        if row['score'] < 0.5:
            continue
        class_name = row['class']
        xmin = row['xmin']
        ymin = row['ymin']
        xmax = row['xmax']
        ymax = row['ymax']
        color = color_mappings.get(class_name, 'yellow')  # Default to yellow if class color not defined
        draw_2.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=2)
        #draw.text((xmin, ymin-15), class_name, fill=color)

    # Display the image with bounding boxes
    col1, col2 = st.columns(2)
    with col1:
        st.text(csv_filename_1)
        st.image(image_1)
    with col2:
        st.text(csv_filename_2)
        st.image(image_2)