import os
import globox
import numpy as np
import pandas as pd
import json
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
    def get_coco_detections(annotation_file: str, filename: str, show_labels=True):
        """
        Create a Decetion object for a given image containing the bounding boxes

        Parameters:
            annotation_file: The file where the COCO annotations are stored
            filename: The image filename
            show_labels: Whether to show the labels or not

        Return:
            Detection object containing the bounding boxes
        """
        coco = globox.AnnotationSet.from_coco(file_path=annotation_file)
        annotations = list(coco)
        annotation = list(filter(lambda x: basename(x.image_id) == basename(filename), annotations))[0]
        bboxes = np.array(list(map(lambda b: [b.xmin, b.ymin, b.xmax, b.ymax], annotation.boxes)))
        class_ids = np.array(list(map(lambda b: 0, annotation.boxes))) if show_labels else None
        return Detection.from_bboxes(bboxes, class_id=class_ids)


    @staticmethod
    def get_mmtracking_detections(annotation_file: str, filename: str, instance_labels=True, show_labels=True):
        """
        Create a Decetion object for a given image containing the bounding boxes

        Parameters:
            annotation_file: The file where the COCO annotations are stored
            filename: The image filename
            instance_labels: Whether to show class or instance IDs
            show_labels: Whether to show the labels or not

        Return:
            Detection object containing the bounding boxes
        """
        with open(annotation_file, 'r') as f:
            content = json.load(f)
        image = list(filter(lambda x: x['file_name'].replace('\\', '/') in filename.replace('\\', '/'), content['images']))[0]
        annotations_raw = list(filter(lambda x: x['image_id'] == image['id'], content['annotations']))
        xyxy = []
        class_id = []
        for box in annotations_raw:
            xyxy.append([
                int(box['bbox'][0]),
                int(box['bbox'][1]),
                int(box['bbox'][0])+int(box['bbox'][2]),
                int(box['bbox'][1])+int(box['bbox'][3])
            ])
            c = box['mot_instance_id'] if instance_labels else box['category_id']
            class_id.append(c)
        class_ids = np.array(class_id) if show_labels else None
        detection = Detection(xyxy=np.array(xyxy), class_id=class_ids, masks=None)
        return detection


    @staticmethod
    def get_trackeval_detections(annotation_dir: str, filename: str, instance_labels=True, show_labels=True):
        """
        Create a Decetion object for a given image containing the bounding boxes

        Parameters:
            annotation_file: The file where the COCO annotations are stored
            filename: The image filename
            instance_labels: Whether to show class or instance IDs
            show_labels: Whether to show the labels or not

        Return:
            Detection object containing the bounding boxes
        """
        annotation_files = os.listdir(annotation_dir)
        for annotation_file in annotation_files:
            if not os.path.splitext(annotation_file)[0] in filename:
                continue
            frame_number = int(os.path.splitext(os.path.basename(filename))[0])
            df = pd.read_csv(f"{annotation_dir}/{annotation_file}", header=None)
            df = df[df[0] == frame_number]
            xyxy = []
            class_id = []
            for i, row in df.iterrows():
                xyxy.append([
                    int(row[2]),
                    int(row[3]),
                    int(row[2])+int(row[4]),
                    int(row[3])+int(row[5])
                ])
                c = int(row[1]) if instance_labels else 1
                class_id.append(c)
            class_ids = np.array(class_id) if show_labels else None
            detection = Detection(xyxy=np.array(xyxy), class_id=class_ids, masks=None)
            return detection


    @staticmethod
    def get_trackformer_detections(annotation_file: str, filename: str, instance_labels=True, show_labels=True):
        """
        Create a Decetion object for a given image containing the bounding boxes

        Parameters:
            annotation_file: The file where the COCO annotations are stored
            filename: The image filename
            instance_labels: Whether to show class or instance IDs
            show_labels: Whether to show the labels or not

        Return:
            Detection object containing the bounding boxes
        """
        with open(annotation_file, 'r') as f:
            content = json.load(f)
        image = list(filter(lambda x: x['file_name'].replace('\\', '/') in filename.replace('\\', '/'), content['images']))[0]
        annotations_raw = list(filter(lambda x: x['image_id'] == image['id'], content['annotations']))
        xyxy = []
        class_id = []
        for box in annotations_raw:
            xyxy.append([
                int(box['bbox'][0]),
                int(box['bbox'][1]),
                int(box['bbox'][0])+int(box['bbox'][2]),
                int(box['bbox'][1])+int(box['bbox'][3])
            ])
            c = box['track_id'] if instance_labels else box['category_id']
            class_id.append(c)
        class_ids = np.array(class_id) if show_labels else None
        detection = Detection(xyxy=np.array(xyxy), class_id=class_ids, masks=None)
        return detection
    

    @staticmethod
    def get_detections(
        annotation_obj: object,
        dataset_name: str,
        filename: str,
        instance_labels=False,
        show_labels=True,
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
            return DatasetAnnotations.get_coco_detections(
                annotation_file=annotation_obj['paths'][0],
                filename=filename,
                show_labels=show_labels,
            )
        elif annotation_obj['type'] == 'mmtracking':
            return DatasetAnnotations.get_mmtracking_detections(
                annotation_file=annotation_obj['paths'][0],
                filename=filename,
                instance_labels=instance_labels,
                show_labels=show_labels,
            )
        elif annotation_obj['type'] == 'trackformer':
            return DatasetAnnotations.get_trackformer_detections(
                annotation_file=annotation_obj['paths'][0],
                filename=filename,
                instance_labels=instance_labels,
                show_labels=show_labels,
            )
        elif annotation_obj['type'] == 'trackeval':
            return DatasetAnnotations.get_trackeval_detections(
                annotation_dir=annotation_obj['paths'][0],
                filename=filename,
                instance_labels=instance_labels,
                show_labels=show_labels,
            )
        elif annotation_obj['type'] == 'masks':
            return DatasetAnnotations.get_mask_detections(
                dataset_name=dataset_name,
                annotation_name=annotation_obj['name'],
                filename=filename,
            )
