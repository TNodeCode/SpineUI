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


def generate_gantt_chart(df):
    fig = ff.create_gantt(df, index_col='Task', bar_width=0.4, show_colorbar=True)
    fig.update_layout(xaxis_type='linear', autosize=False, width=800, height=400)
    return fig

stack_bboxes = [
    np.array([
        [10, 100, 30, 140, 0.77],
    ]),
    np.array([
        [11, 101, 31, 141, 0.79],
        [201, 301, 221, 321, 0.78],
    ]),
    np.array([
        [202, 302, 222, 322, 0.76],
    ]),
    np.array([
        [203, 303, 223, 323, 0.76],
        [400, 400, 420, 420, 0.91],
    ]),
    np.array([
        [204, 304, 224, 324, 0.76],
        [400, 400, 420, 420, 0.91],
    ])
]

tracked_objects = CentroidTracker.stack_tracking(stack_bboxes)

st.title("Object Tracking")

# User has to select a dataset
selected_dataset = st.selectbox("Select Dataset", DatasetConfiguration.get_dataset_names())

if selected_dataset:
    # Get all available stacks in the dataset
    stacks = DatasetConfiguration.get_dataset_stacks(dataset_name=selected_dataset)
    stack_names = stacks.keys()

    # Display a select element for the stacks
    selected_stack = st.selectbox("Select Stack", stack_names)
    stack_entity = stacks[selected_stack]

    # Execute the stack tracking command
    cmd = StackTrackingCommand(dataset_name=selected_dataset, stack_name=selected_stack)
    cmd.execute()   

    # Generate the Gantt chart
    fig = generate_gantt_chart(cmd.df_gantt)

    # Display the Gantt chart using Plotly
    st.plotly_chart(fig, use_container_width=True)