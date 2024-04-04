import streamlit as st
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import json
from src.draw.bboxes import BBoxDrawer
from src.config.datasetconfig import DatasetConfiguration
from src.config.modelconfig import ModelConfiguration
from src.endpoints.endpoint import ModelEndpoint
from src.draw.image_loader import ImageLoader
from src.tracking.tracker import CentroidTracker


df = pd.read_csv("./detections/yolov8x_spine_mixed_val.csv")

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

selected_dataset = st.selectbox("Select Dataset", DatasetConfiguration.get_dataset_names())

if selected_dataset:
    stacks = DatasetConfiguration.get_dataset_stacks(dataset_name=selected_dataset)
    stack_names = stacks.keys()
    selected_stack = st.selectbox("Select Stack", stack_names)

# Read Gantt chart data from a CSV file or create a sample DataFrame
# Replace this with your own data source
df = pd.DataFrame({
    'Task': ['Spine 1', 'Spine 2', 'Spine 3', 'Spine 4', 'Spine 5', 'Spine 6'],
    'Start': ['0', '2', '3', '2', '5', '3'],
    'Finish': ['1', '4', '6', '5', '7', '6']
})

# Generate the Gantt chart
fig = generate_gantt_chart(df)

# Display the Gantt chart using Plotly
st.plotly_chart(fig, use_container_width=True)