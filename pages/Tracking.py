import os
import streamlit as st
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import json
from src.draw.bboxes import BBoxDrawer
from src.commands.tracking import StackTrackingCommand
from src.config.datasetconfig import DatasetConfiguration
from src.config.modelconfig import ModelConfiguration
from src.endpoints.endpoint import ModelEndpoint
from src.draw.image_loader import ImageLoader
from src.tracking.tracker import CentroidTracker
from src.views.comparison.view_tracking_results import view_tracking_results


def generate_gantt_chart(df):
    fig = ff.create_gantt(df, index_col='Task', bar_width=0.4, show_colorbar=True)
    fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)
    return fig

st.title("Object Tracking")

# User has to select a dataset
selected_dataset = st.selectbox("Select Dataset", DatasetConfiguration.get_dataset_names())

if selected_dataset:
    # Get all available stacks in the dataset
    stacks = DatasetConfiguration.get_dataset_stacks(dataset_name=selected_dataset)
    detections = DatasetConfiguration.get_dataset_detections(dataset_name=selected_dataset)
    detections = list(map(lambda x: x['name'], detections))
    stack_names = stacks.keys()

    # Display a select element for the stacks
    selected_stack = st.selectbox("Select Stack", stack_names)
    stack_entity = stacks[selected_stack]

    selected_detections_name = st.selectbox("Select Detections", detections)

    detection_csv_files = DatasetConfiguration.get_detection_csv_files(dataset_name=selected_dataset, detection_name=selected_detections_name)
    selected_detection_file = st.selectbox("Select CSV file", detection_csv_files)

    # Execute the stack tracking command
    cmd = StackTrackingCommand(
        dataset_name=selected_dataset,
        stack_name=selected_stack,
        detections_file=selected_detection_file
    )
    cmd.execute()   

    # Generate the Gantt chart
    fig = generate_gantt_chart(cmd.df_gantt)

    # Display the Gantt chart using Plotly
    st.plotly_chart(fig, use_container_width=True)

    #df_filtered = cmd.traces_df.query(f"stack_id == 0")
    #print("DF", df_filtered)

    view_tracking_results(dataset_name=selected_dataset, traces_df=cmd.traces_df)
