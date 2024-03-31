from typing import List, Optional, Tuple


class TrackingMetrics:
    """
    This class provides methods from computing metrics for 3D object tracking.
    """

    @staticmethod
    def calc_metric_xy(
        centroid1: Optional[List] = None,
        centroid2: Optional[List] = None,
        rect1: Optional[List] = None,
        rect2: Optional[List] = None,
        metric: str = "iom",
    ) -> float:
        """
        calculates metric (usually Intersection over Minimum) between two centroids or two rects

        Args:
            centroid1 (Optional[List]): first centroid in (cX, cY, w, h, ...) format. Defaults to None.
            centroid2 (Optional[List]): second centroid in (cX, cY, w, h, ...) format. Defaults to None.
            rect1 (Optional[List]): first rect in (x1, y1, x2, y2) format. Defaults to None.
            rect2 (Optional[List]): second rect in (x1, y1, x2, y2) format. Defaults to None.
            metric (str): Metric to use, either iou or iom. Defaults to iom

        Raises:
            AttributeError: If neither two centroids nor two boxes are given
            NotImplementedError: metric need to be 'iom' or 'iou', other values are not implemented yet

        Returns:
            float: value of IoM

        This code was copied from:
        https://github.com/SaILaIDiN/Spine-Detection-with-CNNs/blob/main/src/spine_detection/utils/data_utils.py
        """
        # if centroids are given -> calc box coordinates first before calculating everything
        if centroid1 is not None and centroid2 is not None:
            cX1, cY1, w1, h1 = centroid1[:4]
            cX2, cY2, w2, h2 = centroid2[:4]

            x11, x12, y11, y12 = cX1 - w1 / 2, cX1 + w1 / 2, cY1 - h1 / 2, cY1 + h1 / 2
            x21, x22, y21, y22 = cX2 - w2 / 2, cX2 + w2 / 2, cY2 - h2 / 2, cY2 + h2 / 2
        elif rect1 is not None and rect2 is not None:
            x11, y11, x12, y12 = rect1
            x21, y21, x22, y22 = rect2
        else:
            raise AttributeError("You have to provide either two centroids or two boxes but neither is given.")

        area1 = (x12 - x11) * (y12 - y11)
        area2 = (x22 - x21) * (y22 - y21)

        x1, x2, y1, y2 = max(x11, x21), min(x12, x22), max(y11, y21), min(y12, y22)
        if x1 >= x2 or y1 >= y2:
            return 0

        intersection = (x2 - x1) * (y2 - y1)
        union = area1 + area2 - intersection

        if metric == "iom":
            return intersection / min(area1, area2)
        elif metric == "iou":
            return intersection / union
        else:
            raise NotImplementedError(f"Metric {metric} is not implemented")


    @staticmethod
    def calc_metric_z(centroid1: List, centroid2: List) -> float:
        """
        Calculate IoM only for z-direction

        Args:
            centroid1 (List): First centroid of format (..., z1, z2)
            centroid2 (List): Second centroid of format (..., z1, z2)

        Returns:
            float: IoM in z-direction using start and end z-frames z1, z2

        This code was copied from:
        https://github.com/SaILaIDiN/Spine-Detection-with-CNNs/blob/main/src/spine_detection/utils/data_utils.py
        """
        # look how many of centroid1 and centroid2 z-axis overlap
        # using intersection/union, not intersection/minimum
        min_z1, max_z1 = centroid1[-2:]
        min_z2, max_z2 = centroid2[-2:]

        if max_z1 < min_z2 or max_z2 < min_z1:
            return 0

        # +1 has to be added because of how we count with both ends including!
        # if GT is visible in z-layers 5 - 8 (inclusive) and detection is in layer 8 - 9
        # they have one overlap (8), but 8 - 8 = 0 which is wrong!
        intersection = min(max_z1, max_z2) - max(min_z1, min_z2) + 1
        min_val = min(max_z1 - min_z1, max_z2 - min_z2) + 1

        if min_val == 0:
            return 0

        # gt has saved each spine with only one img -04.png
        # should be no problem any more
        return intersection / min_val


    @staticmethod
    def calc_metric(centroid1: List, centroid2: List, metric: str = "iom") -> float:
        """Combine IoM in xy and in z-direction

        Args:
            centroid1 (List): First centroid (cX, cY, w, h, z1, z2)
            centroid2 (List): Second centroid same format
            metric (str): Metric to use, either iou or iom. Defaults to iom

        Returns:
            float: overall F_1-3D-score of both centroids

        This code was copied from:
        https://github.com/SaILaIDiN/Spine-Detection-with-CNNs/blob/main/src/spine_detection/utils/data_utils.py
        """
        # how to combine both metrics
        iom = TrackingMetrics.calc_metric_xy(centroid1, centroid2, metric=metric)
        z_iom = TrackingMetrics.calc_metric_z(centroid1, centroid2)

        # use similar formula to fscore, but replace precision and recall with iom and z_iom
        # beta=low because z_iom should not count that much
        beta = 0.5
        if iom == 0 or z_iom == 0:
            # if iom != 0 and z_iom == 0:
            #     print(f"z-Problem: iom is {iom} while z_iom is {z_iom}")
            return 0
        final_score = (1 + beta**2) * (iom * z_iom) / (beta**2 * iom + z_iom)
        return final_score