import yaml
import glob
import re


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
    def get_dataset_config(dataset_name: str) -> object:
        """
        Get dataset configuration object

        Parameters:
            dataset_name: The name of the dataset in the configuration file

        Return:
            dataset configuration object
        """
        datasets = DatasetConfiguration.get_available_datasets()
        dataset = list(filter(lambda x: x['name'] == dataset_name, datasets))[0]
        return dataset

    
    @staticmethod
    def get_dataset_stacks(dataset_name: str) -> object:
        """
        Get all stacks of a dataset

        Parameters:
            dataset_name: The name of the dataset in the configuration file

        Return:
            An objject containing dataset stack entities
        """
        dataset = DatasetConfiguration.get_dataset_config(dataset_name=dataset_name)
        image_paths = DatasetConfiguration.get_dataset_image_paths(dataset_name=dataset_name)
        dataset_stacks = {}
        for image_path in image_paths:
            for pattern in dataset['stacks']['patterns']:
                # Check if the image path matches the pattern
                matches = re.match(pattern=pattern['regex'], string=image_path)
                # If there is no match we can continue
                if matches is None:
                    continue
                # Get all groups of the regex match
                groups = matches.groups()
                # Build the name of the stack based on the groups
                stack_name = pattern['stack_name']
                for i, group_name in enumerate(groups):
                    stack_name = stack_name.replace("$"+str(i+1), group_name)
                # Create new stack entity if it does not exist
                if stack_name not in dataset_stacks.keys():
                    dataset_stacks |= {stack_name: DatasetStackEntity(name=stack_name, image_paths=[])}
                # Add image path to the stack
                dataset_stacks[stack_name].add_image_path(image_path)
        return dataset_stacks


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


class DatasetStackEntity:
    """
    This class is a data class that represents a stack of images
    """
    def __init__(self, name: str, image_paths: str) -> None:
        """
        Constructor

        Parameters:
            name: stack name
            image_paths: paths of images belonging to this stack
        """
        self.name = name
        self.image_paths = image_paths

    def add_image_path(self, image_path: str):
        """
        Add an image path to the stack

        Parameters:
            image_path: Path of an image
        """
        self.image_paths.append(image_path)
