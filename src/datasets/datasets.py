import os
import glob


class DatasetsHelper:
    DATASETS_DIR = "./datasets"

    """
    This class provides methods for reading datasets
    """
    @staticmethod
    def get_datasets() -> list[str]:
        """
        Looks for all directories in the datasets directory
        
        Return:
            List of all subdirectories in the ./datasets directory
        """
        return os.listdir(DatasetsHelper.DATASETS_DIR)
    
    @staticmethod
    def get_dataset_images(dataset_name: str) -> list[str]:
        """
        Get all images of a dataset

        Parameters:
            dataset_name: the name of the dataset (name of a subdirectory in the datasets directory)

        Return:
            List of all image files in the dataset directory
        """
        imdir = f"{DatasetsHelper.DATASETS_DIR}/{dataset_name}"
        ext = ['png', 'jpg']

        files = []
        [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

        return files
    
    @staticmethod
    def get_dataset_labels(dataset_name: str) -> list[str]:
        """
        Get all images of a dataset

        Parameters:
            dataset_name: the name of the dataset (name of a subdirectory in the datasets directory)

        Return:
            List of all available label files in the dataset directory
        """
        imdir = f"{DatasetsHelper.DATASETS_DIR}/{dataset_name}"
        ext = ['csv']

        files = []
        [files.extend(glob.glob(imdir + '*.' + e)) for e in ext]

        return files