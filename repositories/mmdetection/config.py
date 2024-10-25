import os

def get_classes(filename: str = None) -> "list[str]":
    """
    Get array of class names from text file

    Parameters:
        filename: File where are class names a stored. File has one class name per line.

    Return:
        List of class names
    """
    if filename is None:
        filename = "classes.txt" if not os.getenv("CLASSES_FILE") else os.getenv("CLASSES_FILE")
    with open(filename, "r") as f:
        return f.read().strip().split("\n")


def get_train_annotation_file(default: str = "/data/annotations/instances_train2017.json") -> str:
    """
    Get train annotation file path.

    Parameters:
        default: Default value.

    Return:
        Dataset train annotation file path
    """
    return default if not os.getenv("DATASET_TRAIN_ANNOTATION") else os.getenv("DATASET_TRAIN_ANNOTATION")


def get_train_image_dir(default: str = "/data/train2017") -> str:
    """
    Get train images directory.

    Parameters:
        default: Default value.

    Return:
        Dataset train images directory
    """
    return default if not os.getenv("DATASET_TRAIN_IMAGES") else os.getenv("DATASET_TRAIN_IMAGES")


def get_val_annotation_file(default: str = "/data/annotations/instances_val2017.json") -> str:
    """
    Get validation annotation file path.

    Parameters:
        default: Default value.

    Return:
        Dataset validation annotation file path
    """
    return default if not os.getenv("DATASET_VAL_ANNOTATION") else os.getenv("DATASET_VAL_ANNOTATION")


def get_val_image_dir(default: str = "/data/val2017") -> str:
    """
    Get validation images directory.

    Parameters:
        default: Default value.

    Return:
        Dataset validation images directory
    """
    return default if not os.getenv("DATASET_VAL_IMAGES") else os.getenv("DATASET_VAL_IMAGES")


def get_test_annotation_file(default: str = "/data/annotations/instances_test2017.json") -> str:
    """
    Get test annotation file path.

    Parameters:
        default: Default value.

    Return:
        Dataset test annotation file path
    """
    return default if not os.getenv("DATASET_TEST_ANNOTATION") else os.getenv("DATASET_TEST_ANNOTATION")


def get_test_image_dir(default: str = "/data/test2017") -> str:
    """
    Get test images directory.

    Parameters:
        default: Default value.

    Return:
        Dataset test images directory
    """
    return default if not os.getenv("DATASET_TEST_IMAGES") else os.getenv("DATASET_TEST_IMAGES")
    
    
def get_optimizer_type(default: str = 'SGD') -> str:
    """
    Get optimizer type

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Optimizer type that is used for training
    """
    return default if not os.getenv("OPTIM_TYPE") else os.getenv("OPTIM_TYPE")
    
    
def get_learning_rate(default: float = 0.02) -> float:
    """
    Get learning rate

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Learning rate that is used for training
    """
    return default if not os.getenv("LR") else float(os.getenv("LR"))
    
    
def get_batch_size(default: int = 4) -> int:
    """
    Get batch size

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Batch size that is used for training
    """
    return default if not os.getenv("BATCH_SIZE") else int(os.getenv("BATCH_SIZE"))
    
    
def get_log_interval(default: int = 5) -> int:
    """
    Get log interval

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Interval for printing log messages to the console
    """
    return default if not os.getenv("LOG_INTERVAL") else int(os.getenv("LOG_INTERVAL"))
    
    
def get_number_of_epochs(default: int = 12) -> int:
    """
    Get number of epochs

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Number of epochs that the model will be trained
    """
    return default if not os.getenv("EPOCHS") else int(os.getenv("EPOCHS"))
    
    
def get_workers_per_gpu(default: int = 4) -> int:
    """
    Get workers per GPU

    Parameters:
        default: Default value if no environment variable is set

    Return:
        Workers per GPU
    """
    return default if not os.getenv("WORKERS") else int(os.getenv("WORKERS"))


def get_augmentations():
    """
    List of image augmentations that should be applied during training of the models

    Return:
        List oof image augmentations
    """
    return [
        #dict(type='Corrupt', corruption='brightness', severity=3),
        #dict(type='Corrupt', corruption='contrast', severity=1),
        #dict(type='Corrupt', corruption='elastic_transform', severity=1),
        #dict(type='Corrupt', corruption='gaussian_noise', severity=2),
        #dict(type='Corrupt', corruption='gaussian_blur', severity=1),
        #dict(type='Corrupt', corruption='pixelate', severity=1),
        dict(type='RandomFlip', flip_ratio=0.5, direction="horizontal"),
        dict(type='RandomFlip', flip_ratio=0.5, direction="vertical"),
        dict(type='RandomFlip', flip_ratio=0.5, direction="diagonal"),
        dict(type='RandomAffine', max_rotate_degree=10.0, max_translate_ratio=0.1, scaling_ratio_range=(0.5, 1.5), max_shear_degree=2.0),
    ]

def get_detector_checkpoint(default: str):
    """
    Get a checkpoint for the detector model for inference

    Parameters:
        default: Default value if no environment variable is set

    Return:
        checkpoint filename
    """
    return os.getenv("CKP_DETECTOR") if os.getenv("CKP_DETECTOR") else default


def get_reid_checkpoint(default: str):
    """
    Get a checkpoint for the reid model for inference

    Parameters:
        default: Default value if no environment variable is set

    Return:
        checkpoint filename
    """
    return os.getenv("CKP_REID") if os.getenv("CKP_REID") else default