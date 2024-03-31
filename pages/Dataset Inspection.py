import streamlit as st
from src.views.comparison.view_inspection import view_dataset_inspection

# headline
st.markdown("# Dataset Inspection")

view_dataset_inspection()

