import streamlit as st
import pandas as pd

from src.config.datasetconfig import DatasetConfiguration


st.markdown("# Dataset Management")
st.text("On this site you can manage your datasets")

datasets = []
for dataset in DatasetConfiguration.get_available_datasets():
    paths = DatasetConfiguration.get_dataset_image_paths(dataset['name'])
    datasets.append({"name": dataset['name'], "images": len(paths)})

st.dataframe(pd.DataFrame(datasets))

