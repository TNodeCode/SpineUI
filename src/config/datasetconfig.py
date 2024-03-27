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