from globox import AnnotationSet, Annotation, BoundingBox, COCOEvaluator
import pandas as pd
import glob
import re


class IoMinBoundingBox(BoundingBox):
    """
    This class overwrites the iou method from the BoundingBox class.
    It computes the intersection overminimum when "iou" is called.
    """
    def iou(self, other: BoundingBox) -> float:
        """
        The Intersection over Minimum (IoM) between two bounding boxes.
        
        Parameters:
            other: Other bounding box

        Return:
            Intersection over minimum between this and other bounding box
        """
        xmin = max(self._xmin, other._xmin)
        ymin = max(self._ymin, other._ymin)
        xmax = min(self._xmax, other._xmax)
        ymax = min(self._ymax, other._ymax)

        if xmax < xmin or ymax < ymin:
            return 0.0

        intersection = (xmax - xmin) * (ymax - ymin)
        minimum = min(self.area, other.area)

        if minimum == 0.0:
            return 1.0

        return min(intersection / minimum, 1.0)


def get_coco_gt_annotation_set(
        file_path: str,
        filename_pattern: str = None,
) -> AnnotationSet:
    _gt_annotations = AnnotationSet.from_coco(file_path=file_path)
    gt_annotations = AnnotationSet()
    for ann in _gt_annotations:
        # If a filename pattern is provided we check if the annotation image filenames match this pattern
        if filename_pattern:
            if re.match(filename_pattern, ann.image_id):
                gt_annotations.add(ann)
            else:
                continue
        # Otherwise the images are added without checks
        else: 
            gt_annotations.add(ann)
    return gt_annotations


def read_csv_bboxes_as_df(csv_filepath: str) -> pd.DataFrame:
    return pd.read_csv(csv_filepath)


def bboxes_df_to_annotation_set(
        df: pd.DataFrame,
        threshold: float = 0.5,
        filename_pattern: str = None,
        bbox_class = BoundingBox, 
) -> AnnotationSet:
    pred_annotations = AnnotationSet()
    df = df[df["score"] >= threshold]
    if filename_pattern:
        df = df[df["filename"].str.match(filename_pattern)]
    filenames = df["filename"].drop_duplicates()

    # Iterate over all filenames in the data frame
    for filename in filenames:
        # get all bounding boxes for a given image file
        bboxes = df[df["filename"] == filename]
        boxes = []
        for i, row in bboxes.iterrows():
            # transform bounding boxes into the globox format
            boxes.append(bbox_class(
                label=str(row["class_name"]),
                xmin=int(row["xmin"]),
                ymin=int(row["ymin"]),
                xmax=int(row["xmax"]),
                ymax=int(row["ymax"]),
                confidence=float(row["score"])
            ))
        # TODO compute image sizes
        annotation = Annotation(image_id=filename, image_size=(512,512), boxes=boxes)
        pred_annotations.add(annotation=annotation)
    return pred_annotations



def evaluate_predicted_bboxes(
        gt_annotations: AnnotationSet,
        pred_annotations: AnnotationSet,
        threshold: float = 0.5,
        max_detections: int = 100,
) -> dict:
    evaluator = COCOEvaluator(
        ground_truths=gt_annotations, 
        predictions=pred_annotations
    )
    evaluation = evaluator.evaluate(
        iou_threshold=threshold,
        max_detections=max_detections
    )

    ap = evaluation.ap()
    ar = evaluation.ar()
    f1 = (2*ap*ar)/(ap+ar) if ap+ar > 0.0 else 0.0
    metrics = {
        "ap": ap,
        "ar": ar,
        "f1": f1,
        "ap_50": evaluator.ap_50(),
        "ap_75": evaluator.ap_75(),
        "ap_small": evaluator.ap_small() if evaluator.ap_small() else 0.0,
        "ap_medium": evaluator.ap_medium() if evaluator.ap_medium() else 0.0,
        "ap_large": evaluator.ap_large() if evaluator.ap_large() else 0.0,
        "ar_1": evaluator.ar_1(),
        "ar_10": evaluator.ar_10(),
        "ar_100": evaluator.ar_100(),
        "ar_small": evaluator.ar_small() if evaluator.ar_small() else 0.0,
        "ar_medium": evaluator.ar_medium() if evaluator.ar_medium() else 0.0,
        "ar_large": evaluator.ar_large() if evaluator.ar_large() else 0.0,
    }
    return metrics


def evaluate_training_epochs(
        gt_file_path: str,
        model_epoch_filename_func,
        max_epochs: int,
        annotation_filename_pattern = None,
):
    # In this list we store the metrics of all epochs
    epoch_metrics = []

    # These are the ground truth annotations the predictions are evaluated against
    gt_annotations = get_coco_gt_annotation_set(
        file_path=gt_file_path,
        filename_pattern=annotation_filename_pattern,
    )

    for i in range(1, max_epochs+1):
        # Build the filename of the file where bboxes of i-th epoch can be found
        filename = model_epoch_filename_func(i)
        # Read this CSV file
        df = read_csv_bboxes_as_df(filename)
        # Create an annotation set based on that CSV file
        pred_annotations = bboxes_df_to_annotation_set(
            df=df,
            bbox_class=IoMinBoundingBox,
            filename_pattern=annotation_filename_pattern
        )
        # Compute the metrics for the i-th epoch
        metrics = evaluate_predicted_bboxes(
            gt_annotations=gt_annotations,
            pred_annotations=pred_annotations
        )
        # Add metrics of i-th epoch to list
        epoch_metrics.append(metrics)
    return pd.DataFrame(epoch_metrics)

trained_on_dict={
    "altug": "altug_",
    "mixed": "",
    "simon": "simon_"
}
evaluated_on_dict={
    "altug": ".*aidv.*",
    "mixed": None,
    "simon": ".*SR052.*",
}

dataset = "test"
for trained_on in ("mixed", "simon", "altug"):
    for evaluated_on in ("mixed", "simon", "altug"):
        df_metrics = evaluate_training_epochs(
            gt_file_path="./datasets/spine/annotations/instances_val2017.json",
            model_epoch_filename_func=lambda i: f"C:/Users/tilof/Downloads/co_detr_evaluation/co_detr_co_dino_5scale_swin_large_16e_o365tococo_spine_{trained_on_dict[trained_on]}epoch_{i}_{dataset}2017.csv",
            max_epochs=25,
            annotation_filename_pattern=evaluated_on_dict[evaluated_on]
        )
        results_filename=f"co_detr_trained_on_{trained_on}_evaluated_on_{evaluated_on}_{dataset}_data.csv"
        df_metrics.to_csv(results_filename, index=False)