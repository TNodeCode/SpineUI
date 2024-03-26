import yaml


class ModelConfiguration:
    CONFIG_FILE = 'config/models.yml'

    @staticmethod
    def read_config_file():
        with open(ModelConfiguration.CONFIG_FILE, 'r') as file:
            config = yaml.safe_load(file)
        return config

    @staticmethod
    def get_available_endpoints():
        config = ModelConfiguration.read_config_file()
        return config['models']
