import os
import pandas as pd
from src.config.datasetconfig import DatasetConfiguration
from src.tracking.tracker import CentroidTracker


class StackTrackingCommand():
    """
    Command class that performs object tracking on a dataset
    """
    def __init__(self, dataset_name: str, stack_name: str):
        """
        Constructor

        Args:
            dataset_name: Name of the dataset the tracking should be performed on
            stack_name: Name of the stack
        """
        self.executed = False
        self.dataset_name = dataset_name
        self.stack_name = stack_name
        self.tracking_results = None
        self.df_gantt = None
        self.traces_df = None


    def execute(self):
        """
        Execute tracking on the given dataset and stack
        """
        if self.executed:
            return
        
        # Get all available stacks in the dataset
        stacks = DatasetConfiguration.get_dataset_stacks(dataset_name=self.dataset_name)

        # Display a select element for the stacks
        stack_entity = stacks[self.stack_name]

        # Read all detected bounding boxes
        df_det = pd.read_csv("./detections/faster_rcnn/faster_rcnn_run1_spine_mixed_train.csv") # TODO
        stack_bboxes = []
        for filename in stack_entity.image_paths:
            bboxes = df_det[df_det['filename'] == os.path.basename(filename)][["xmin", "ymin", "xmax", "ymax", "score"]].to_numpy()
            bboxes[:, 0:3] = bboxes[:, 0:3].astype(int)
            stack_bboxes.append(bboxes)
        
        # Run bounding boxes through tracking algorithm
        self.tracking_results = CentroidTracker.stack_tracking(stack_bboxes=stack_bboxes)

        # Create a Gantt chart datafarame from the tracking results
        self.df_gantt = CentroidTracker.generate_gantt_df(self.tracking_results)

        # Create a dataframe containing all bounding boxes with object ids and frame ids
        self.traces_df = CentroidTracker.get_traces_df(
            tracking_results=self.tracking_results,
            image_paths=stack_entity.image_paths
        )

        self.executed = True

