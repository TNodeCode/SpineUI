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

# headline
st.markdown("# Object Detection Model Evaluation and Comparison")

view_model_comparison()