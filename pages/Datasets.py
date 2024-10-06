import streamlit as st
import pandas as pd
import globox

from src.config.datasetconfig import DatasetConfiguration


st.markdown("# Dataset Management")
st.text("On this site you can manage your datasets")

datasets = []
for dataset in DatasetConfiguration.get_available_datasets():
    paths = DatasetConfiguration.get_dataset_image_paths(dataset['name'])
    annotations = DatasetConfiguration.get_dataset_annotations(dataset['name'])
    for ann in annotations:
        if ann['type'] in ['coco', 'mmtracking']:
            coco: globox.AnnotationSet = globox.AnnotationSet.from_coco(file_path=ann['paths'][0])
            n_boxes = len(list(coco.all_boxes))
    datasets.append({"name": dataset['name'], "images": len(paths), "objects": n_boxes})

st.dataframe(pd.DataFrame(datasets))

