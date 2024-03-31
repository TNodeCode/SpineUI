from src.util.detection import Detection
from PIL import Image
import numpy as np


class Masks:
    """
    This class provides methods for handling masks
    """
    @staticmethod
    def mask_images_to_detection(mask_image_paths: list[str]):
        """
        Load mask images and turn them into boolean matrices

        Parameters:
            mask_image_paths: Paths where the mask images can be found

        Return:
            array of 2D boolean matrices representing masks
        """
        masks = []
        for path in mask_image_paths:
            mask = np.array(Image.open(path))
            masks.append(mask)
        return Detection.from_masks(masks=np.array(masks).astype(bool))