import numpy as np
import supervision as sv
from PIL import Image


class Detection:
    """
    This class is a data class represents all bounding boxes / masks found in an image.
    It can be used for image annotation with the Roboflow supervision library.
    """
    def __init__(self, xyxy: np.ndarray, masks: np.ndarray, class_id: np.ndarray):
        """
        Constructor

        Parameters:
            xyxy: A list of bounding boxes in xyxy format
            masks: An array of 2D boolean matrices
            class_id: An array of integers representing class IDs
        """
        # list of bounding boxes in xyxy format
        self.xyxy = xyxy
        # array of 2D boolean arrays rrpesenting masks
        self.mask = masks
        if self.mask is not None:
            # 1D integer array representing classes
            self.class_id = np.array(list(map(lambda mask: 0, self.mask)))
        elif self.xyxy is not None:
            self.class_id = np.zeros(self.xyxy.shape[0]).astype(int)
        else:
            raise AssertionError("Either bounding boxes or masks must be provided")
        # 1D integer array representing areas of bounding boxes
        self.area = np.array(list(map(lambda xyxy: (xyxy[2]-xyxy[0])*(xyxy[3]-xyxy[1]), self.xyxy)))

    
    @staticmethod
    def from_bboxes(bboxes: np.ndarray, class_id: np.ndarray = None):
        """
        Build a detection object based on an array of bounding boxes

        Parameters:
            bboxes: A 2D array of bounding boxes in xyxy format
            class_id: An array of integers representing class IDs

        Return:
            Detection object
        """
        assert type(bboxes) == np.ndarray, "Bounding boxes must be a numpy array"
        assert len(bboxes.shape) == 2, "Bounding boxes must be a 2D array"
        return Detection(xyxy=bboxes, masks=None, class_id=class_id)


    @staticmethod
    def from_masks(masks: np.ndarray, class_id: np.ndarray=None):
        """
        Build a detection object based on an array of 2D boolean mask items

        Parameters:
            masks: An array of 2D boolean matrices
            class_id: An array of integers representing class IDs

        Return:
            Detection object
        """
        assert type(masks) == np.ndarray, "Masks must be a numpy array"
        assert len(masks.shape) == 3, "Masks must be a 3D array"
        xyxy = np.array(list(map(lambda mask: Detection.mask_to_bbox(mask), masks)))
        if class_id is None:
            class_id = np.array(list(map(lambda mask: 0, masks)))
        return Detection(xyxy=xyxy, masks=masks, class_id=class_id)
        
    @staticmethod
    def mask_to_bbox(mask):
        """
        Convert a mask to a bounding box by finding the minimum and maximum coordinates in each dimension

        Parameters:
            mask: A 2D boolean matrix

        Return:
            Bounding box in xyxy format
        """
        positive_cols = np.where(np.any(mask, axis=0))[0]
        positive_rows = np.where(np.any(mask, axis=1))[0]
        min_x, max_x, min_y, max_y = positive_cols.min(), positive_cols.max(), positive_rows.min(), positive_rows.max()
        return min_x, min_y, max_x, max_y
    

    @staticmethod
    def plot_detections(image: Image, detections, show_masks=True, show_bboxes=True):
        """
        Plot bounding boxes and segmentation masks

        Parameters:
            image: PIL Image object
            detections: Detection object
            show_masks: Whether masks should be plottet
            show_bboxes: Whether bounding boxes should be plottet
        """
        mask_annotator = sv.MaskAnnotator()
        box_annotator = sv.BoxAnnotator()
        if show_masks:
            image = mask_annotator.annotate(
                scene=image.copy(),
                detections=detections,
            )
        if show_bboxes:
            image = box_annotator.annotate(
                scene=image.copy(),
                detections=detections,
            )
        return image
    
    def __len__(self):
        """
        Get number of boxes / masks

        Return:
            Number of bxes / masks
        """
        return len(self.xyxy)