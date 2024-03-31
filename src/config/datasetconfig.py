import yaml
import glob


class DatasetConfiguration:
    """
    This class provides methods for accessing the dataset configuration file.
    """
    CONFIG_FILE = 'config/datasets.yml'

    @staticmethod
    def read_config_file() -> dict:
        """
        Read the dataset configuration file

        Return: 
            Dataset configuration object
        """
        with open(DatasetConfiguration.CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def get_available_datasets() -> list[dict]:
        """
        Get a list of all available datasets

        Return:
            Configuration object of all available datasets
        """
        config = DatasetConfiguration.read_config_file()
        return config['datasets']

    @staticmethod
    def get_dataset_names() -> list[str]:
        """
        Get a list of all available dataset names

        Return:
            Names of all available datasets
        """
        config = DatasetConfiguration.read_config_file()
        return list(map(lambda x: x['name'], config['datasets']))

    @staticmethod
    def get_dataset_image_paths(dataset_name: str) -> list[str]:
        """
        Get paths of all images belonging to a dataset

        Parameters:
            dataset_name: The name of the dataset in the configuration file

        Return:
            A list of all image paths belonging to the dataset
        """
        datasets = DatasetConfiguration.get_available_datasets()
        dataset = list(filter(lambda x: x['name'] == dataset_name, datasets))[0]
        image_paths = []
        for path in dataset['paths']:
            image_paths = image_paths + glob.glob(path)
        return image_paths
    

    @staticmethod
    def get_dataset_annotations(dataset_name: str) -> list[object]:
        """
        Get annotation configurations of a dataset

        Parameters:
            dataset_name: The name of the dataset in the configuration file

        Return:
            A list of all annotation configurations
        """
        datasets = DatasetConfiguration.get_available_datasets()
        dataset = list(filter(lambda x: x['name'] == dataset_name, datasets))[0]
        return dataset['annotations']
    

    @staticmethod
    def get_dataset_annotation(dataset_name: str, annotation_name: str) -> object:
        """
        Get annotation configurations of a dataset

        Parameters:
            dataset_name: The name of the dataset in the configuration file

        Return:
            A list of all annotation configurations
        """
        annotations = DatasetConfiguration.get_dataset_annotations(dataset_name=dataset_name)
        annotation_obj = list(filter(lambda x: x['name'] == annotation_name, annotations))[0]
        return annotation_obj
    

    @staticmethod
    def get_dataset_mask_annotations(dataset_name: str) -> list[object]:
        """
        Get mask annotation configurations of a dataset

        Parameters:
            dataset_name: The name of the dataset in the configuration file
            annotation_name: The name of the annotation object in the configutation file

        Return:
            A list of all mask annotation configurations
        """
        annotations = DatasetConfiguration.get_dataset_annotations(dataset_name=dataset_name)
        return list(filter(lambda x: x['type'] == 'masks', annotations))
    

    @staticmethod
    def get_dataset_mask_image_paths(dataset_name: str, annotation_name: str) -> list[str]:
        """
        Get all mask images for a given dataset and annotation name

        Parameters:
            dataset_name: The name of the dataset in the configuration file
            annotation_name: The name of the annotation object in the configutation file

        Return:
            A list of all mask images
        """
        annotations = DatasetConfiguration.get_dataset_mask_annotations(dataset_name=dataset_name)
        annotation_obj = list(filter(lambda x: x['name'] == annotation_name, annotations))[0]
        image_paths = []
        for path in annotation_obj['paths']:
            image_paths = image_paths + glob.glob(path)
        return image_paths
