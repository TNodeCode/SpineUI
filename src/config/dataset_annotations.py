import globox
import numpy as np
from os.path import basename
from src.config.datasetconfig import DatasetConfiguration
from src.datasets.masks import Masks
from src.util.detection import Detection


class DatasetAnnotations:
    """
    This class provides helper method for loading annotations that are defined
    in the dataset configuration file
    """

    @staticmethod
    def get_mask_detections(dataset_name: str, annotation_name: str, filename: str):
        """
        Get array of 2D segmentation masks for an image file

        Parameters:
            dataset_name: The name of the dataset the image belongs to
            annotatation_name: The name of the annotation file
            filename: The name of the image file

        Return:
            A Detectioon object containing 2D boolean segmentation masks that belong to the given image
        """
        # Get all mask image paths that belong to a dataset
        mask_images = DatasetConfiguration.get_dataset_mask_image_paths(
            dataset_name=dataset_name,
            annotation_name=annotation_name
        )
        # Get mask image paths
        mask_image_paths = list(filter(lambda x: basename(filename) in x, mask_images))
        # Create detection objects
        detections = Masks.mask_images_to_detection(mask_image_paths)
        return detections


    @staticmethod
    def get_coco_detections(annotation_file: str, filename: str):
        """
        Create a Decetion object for a given image containing the bounding boxes

        Parameters:
            annotation_file: The file where the COCO annotations are stored
            filename: The image filename

        Return:
            Detection object containing the bounding boxes
        """
        coco = globox.AnnotationSet.from_coco(file_path=annotation_file)
        annotations = list(coco)
        annotation = list(filter(lambda x: basename(x.image_id) == basename(filename), annotations))[0]
        bboxes = np.array(list(map(lambda b: [b.xmin, b.ymin, b.xmax, b.ymax], annotation.boxes)))
        return Detection.from_bboxes(bboxes)


    @staticmethod
    def get_detections(
        annotation_obj: object,
        dataset_name: str,
        filename: str
    ):
        """
        Get dataset detections

        Parameters:
            annotation_obj: Annotation object from the configuration file
            dataset_name: Name of the dataset
            filename: Name of the image file

        Return:
            Detection object containing either bounding boxes or segmentation masks
        """
        if annotation_obj['type'] == 'coco':
            return DatasetAnnotations.get_coco_detections(annotation_file=annotation_obj['paths'][0], filename=filename)
        elif annotation_obj['type'] == 'masks':
            return DatasetAnnotations.get_mask_detections(dataset_name=dataset_name, annotation_name=annotation_obj['name'], filename=filename)
