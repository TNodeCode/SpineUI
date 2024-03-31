import streamlit as st
import numpy as np


class Pagination:
    def __init__(self, df) -> None:
        self.df = df
        if 'selected_index' not in st.session_state:
            st.session_state['selected_index'] = 0
        self.selected_index = st.session_state['selected_index']
        self.num_filenames = len(df['filename'].unique())
        self.previous_index = (self.selected_index - 1) % self.num_filenames
        self.next_index = (self.selected_index + 1) % self.num_filenames

    def forward(self):
        self.selected_index = self.next_index
        st.session_state['selected_index'] = self.selected_index
        self.previous_index = (self.selected_index - 1) % self.num_filenames
        self.next_index = (self.selected_index + 1) % self.num_filenames

    def backward(self):
        self.selected_index = self.previous_index
        st.session_state['selected_index'] = self.selected_index
        self.previous_index = (self.selected_index - 1) % self.num_filenames
        self.next_index = (self.selected_index + 1) % self.num_filenames

    def update_selected_index(self, filename):
        self.selected_index = int(np.where(self.df['filename'].unique() == filename)[0][0])
        st.session_state['selected_index'] = self.selected_index
        self.previous_index = (self.selected_index - 1) % self.num_filenames
        self.next_index = (self.selected_index + 1) % self.num_filenames
