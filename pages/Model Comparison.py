import streamlit as st
import os
import pandas as pd
import numpy as np
import glob
from PIL import Image, ImageDraw
from src.views.comparison.view_comparison import view_model_comparison

# Create dataset directory if it does not exist
if not os.path.isdir("./datasets"):
    os.makedirs("./datasets", exist_ok=True)

# Base path where to look for the images
basepath = "./datasets/spine/val"
available_cvs_files = glob.glob("detections/*.csv")

# headline
st.markdown("# Object Detection Model Evaluation and Comparison")

if len(available_cvs_files) < 1:
    st.warning('No CSV detection result files found', icon="⚠️")
else:
    view_model_comparison(
        basepath=basepath,
        cvs_files=available_cvs_files
    )