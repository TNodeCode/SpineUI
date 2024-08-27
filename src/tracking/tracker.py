from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance as dist

from src.tracking.metrics import TrackingMetrics


class CentroidTracker:
    """
    Control everything for tracking the spines
    
    The code of this class was copied from:
    https://github.com/SaILaIDiN/Spine-Detection-with-CNNs/blob/main/src/spine_detection/utils/tracker.py
    """

    def __init__(
        self,
        maxDisappeared: int = 50,
        minAppeared: int = 50,
        maxDiff: int = 30,
        maxVol: int = 80 * 80,
        iomThresh: float = 0.7,     # value in the range 0.0 < iomThresh <= 1.0
        metric: str = "iom",
    ) -> None:
        """Initialize parameters and the tracker

        Args:
            maxDisappeared (int, optional): how many frames an object isn't seen before counting as disappeared
                Defaults to 50.
            minAppeared (int, optional): how many frames are needed to detect an object as real. Defaults to 50.
            maxDiff (int, optional): max pixel difference for being identified as same object. Defaults to 30.
            maxVol (int, optional): max volume of spines allowed to count as object. Defaults to 80*80.
            iomThresh (float, optional): min IoM necessary to identify as the same spine. Defaults to 0.7.
            metric (str, optional): Which metric should be used, currently 'iom' or 'iou' available. Defaults to 'iom'.
        """
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.object_traces = OrderedDict()
        self.beforeObjects = OrderedDict()
        self.disappeared = OrderedDict()
        self.appeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared
        self.minAppeared = minAppeared
        self.maxDiff = maxDiff
        self.maxVol = maxVol
        self.iomThresh = iomThresh

        if type(metric) == str:
            self.metric = lambda centroid1, centroid2: TrackingMetrics.calc_metric_xy(centroid1, centroid2, metric=metric)
        else:
            self.metric = metric

    def register(self, centroid: np.ndarray) -> None:
        """register a new centroid

        Args:
            centroid (np.ndarray): Centroid in [cX, cY, w, h] format
        """
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.object_traces[self.nextObjectID] = [centroid]
        self.disappeared[self.nextObjectID] = 0
        self.appeared[self.nextObjectID] = 1
        self.nextObjectID += 1

    def getObjects(self) -> OrderedDict:
        """Get all objects that are not hidden

        Returns:
            OrderedDict: dict with id, centroid pairs
        """
        correctObjects = OrderedDict()
        for objectID in self.objects.keys():
            if self.appeared[objectID] > self.minAppeared and self.disappeared[objectID] <= self.maxDisappeared:
                correctObjects[objectID] = self.objects[objectID]
        return correctObjects

    def preprocess(self, inputCentroids: np.ndarray) -> List[List]:
        """Preprocess np array of centroids to only interesting centroids

        Args:
            inputCentroids (np.ndarray): centroids of detection

        Returns:
            List[List]: List of centroids in (cX, cY, w, h) format
        """
        # input must be a np array
        # delete all centroids which are in each other and have the lower probability
        deleted_index = []
        added_centroid = []
        for i in range(len(inputCentroids)):
            # find the box with highest iom
            max_iom = self.iomThresh
            max_index = -1
            for j in range(i + 1, len(inputCentroids)):
                if j in deleted_index:
                    continue
                iom = self.metric(inputCentroids[i], inputCentroids[j])
                if iom >= max_iom:
                    max_iom = iom
                    max_index = j
            if max_index == -1:
                continue
            else:
                if inputCentroids[i][4] >= inputCentroids[max_index][4]:
                    deleted_index.append(max_index)
                else:
                    deleted_index.append(i)

        if len(added_centroid) != 0:
            inputCentroids = np.concatenate((inputCentroids, added_centroid), axis=0)
        ret_list = [inputCentroids[i] for i in range(len(inputCentroids)) if i not in deleted_index]
        return ret_list

    def update(self, rects: List[List]) -> OrderedDict:
        """Update tracker with detection rects

        Args:
            rects (List[List]): List of rects in format (x1, y1, x2, y2, conf)

        Returns:
            OrderedDict: dict with id, centroid pairs
        """
        # check to see if the list of input bounding box rectangles is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them as disappeared
            for objectID in self.objects.keys():
                self.disappeared[objectID] += 1

            # return early as there are no centroids or tracking info to update
            return self.getObjects()

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 5), dtype="float")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY, conf)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            w = int(endX - startX)
            h = int(endY - startY)
            if w * h >= self.maxVol:
                continue
            inputCentroids[i] = (cX, cY, w, h, conf)

        inputCentroids = self.preprocess(inputCentroids)

        # if we are currently not tracking any objects take the input centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, try to match the input centroids to existing object centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object centroids and input centroids, respectively
            # don't use distance, but IoM as comparison -> min is replaced by max!
            D = dist.cdist(np.array(objectCentroids), inputCentroids, metric=self.metric)

            # row -> original tracked objects, cols -> new input bboxes
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            rows = D.max(axis=1).argsort()[::-1]

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmax(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # is the distance too big, so we have to register a new object!
                if D[row, col] <= self.maxDiff:  # > self.maxDiff:
                    objectID = objectIDs[row]
                    self.register(inputCentroids[col])
                    self.disappeared[objectID] += 1
                else:
                    # otherwise, grab the object ID for the current row,
                    # set its new centroid, and reset the disappeared
                    objectID = objectIDs[row]
                    self.objects[objectID] = inputCentroids[col]
                    self.object_traces[objectID].append(inputCentroids[col])
                    self.appeared[objectID] += 1
                    self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # compare the number of inputCentroids and the number of real
            # existing centroids (not this ones with appeared <= minAppeared)
            object_len = len([1 for key in self.appeared.keys() if self.appeared[key] > self.minAppeared])

            if object_len >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

            # register the input centroids as new objects in any case
            for col in unusedCols:
                self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.getObjects()
    
    @staticmethod
    def stack_tracking(
        stack_bboxes: np.ndarray,
        minAppeared: int = 1,
        maxDisappeared: int = 1,
        maxDiff=0.7,
        iomThresh=0.3,
    ):
        """
        Perform tracking on a stack of images

        Args:
            stack_bboxes: 3D array of bounding boxes
                          first dim = images
                          second dim = list of bounding boxes per image
                          third dim = single bounding box in the format [x1, y1, x2, y2, conf]
            minAppeared: minimum number of appearances that an object needs to be recognized as an object 
            maxDisappeared: maximum number of frames that an object is allowed to disappear
            maxDiff: max pixel difference for being identified as same object
            iomThresh: Maximum IoM threshold between to frames for an object

        Return:
            A dictionary containing all objects with the first and last frame they appeared in
        """
        # TODO add arguments to method so that the user can chage these parameters
        ct = CentroidTracker(
            minAppeared=minAppeared,
            maxDisappeared=maxDisappeared,
            maxDiff=maxDiff,
            iomThresh=iomThresh,
        )

        # Save in which frame objects wre first and last seen
        objects_first_appearance = OrderedDict()
        objects_last_appearance = OrderedDict()

        # Iterate over images
        for frame, single_image_bboxes in enumerate(stack_bboxes):
            # Pass bounding boxes of i-th frame to the tracker
            ct.update(single_image_bboxes)
            for object_id in ct.appeared.keys():
                if object_id not in objects_first_appearance.keys():
                    objects_first_appearance[object_id] = frame

        for object_id in ct.objects.keys():
            objects_last_appearance[object_id] = objects_first_appearance[object_id] + ct.appeared[object_id] - 1

        data =  {
            "object_ids": ct.objects.keys(),
            "object_traces": ct.object_traces,
            "first_appearance": objects_first_appearance,
            "last_appearance": objects_last_appearance,
        }

        return data
    

    @staticmethod
    def generate_gantt_df(tracking_results: dict):
        """
        Convert tracking results into a pandas dataframe which can be displayed as a Gantt chart

        Args:
            tracking_results: the results of the tracking algorithm

        Return:
            A pandas dataframe containing task names, strat and finish points
        """
        df_gantt = pd.DataFrame({
            'Task': list(tracking_results['object_ids']),
            'Start': list(map(lambda x: x[1], tracking_results['first_appearance'].items())),
            'Finish': list(map(lambda x: x[1], tracking_results['last_appearance'].items()))
        })
        return df_gantt

    
    @staticmethod
    def get_traces_df(tracking_results: dict, image_paths: list = None):
        """
        Turn Gantt dataframe into traces dataframe

        Args:
            tracking_results: the results of the tracking algorithm

        Return:
            Dataframe containing bounding boxes with tracking IDs
        """
        bboxes = []

        # Iterate over all detected objects
        for id in tracking_results['object_ids']:
            fap = tracking_results['first_appearance'][id]
            # Iterate over all bounding boxes that belong to an object
            for i, bbox in enumerate(tracking_results['object_traces'][id]):
                frame = fap+i
                bbox = {
                    'frame': int(frame),
                    'object_id': int(id),
                    'x0': int(bbox[0] - 0.5 * bbox[2]),
                    'y0': int(bbox[1] - 0.5 * bbox[3]),
                    'w': int(bbox[2]),
                    'h': int(bbox[3]),
                    'score': float(bbox[4]),
                }
                if image_paths is not None:
                    bbox |= {'filename': image_paths[frame]}
                bboxes.append(bbox)

        # Create dataframe
        df = pd.DataFrame(bboxes)
        if df.shape[0] > 0:
            convert_dict = {
                'object_id': int,
                'frame': int,
                'x0': int,
                'y0': int,
                'w': int,
                'h': int,
            }
            df = df.astype(convert_dict)
        return df
        







