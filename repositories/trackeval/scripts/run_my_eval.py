import trackeval  # Import trackeval library

class CustomMotDataset(trackeval.datasets.MotChallenge2DBox):
    def __init__(self, config=None):
        super(CustomMotDataset, self).__init__(config=config)
        self.valid_class_numbers.extend([0])


# Define paths and other configurations
eval_config = {
    'USE_PARALLEL': False,          # Whether to use parallel processing
    'NUM_PARALLEL_CORES': 8,        # Number of cores to use if parallel processing is enabled
    'PRINT_RESULTS': True,          # Whether to print the results
    'PRINT_CONFIG': True,           # Whether to print the configuration
    'TIME_PROGRESS': True,          # Whether to display the time progress
    'DISPLAY_LESS_PROGRESS': False, # Whether to display less progress
    'OUTPUT_SUMMARY': True,         # Whether to output a summary of the results
    'OUTPUT_DETAILED': True,        # Whether to output detailed results
    'PLOT_CURVES': False,           # Whether to plot curves
}

dataset_config = {
    'TRACKERS_FOLDER': '../mmtracking/data/MOT17-tracks',  # Folder containing the tracker results
    'GT_FOLDER': '../mmtracking/data/MOT17/train',              # Folder containing the ground truth
    'SEQMAP_FILE': '../mmtracking/data/MOT17/train/seqmaps.txt',          # File containing the sequence map
    'OUTPUT_FOLDER': 'results',  # Where to save the outputs (if None, nothing is saved)
    'TRACKER_SUB_FOLDER': '',   # Subfolder within the tracker folder containing the data
    'BENCHMARK': 'MOT17',            # Which benchmark to use (MOT17, KITTI, etc.)
    'SEQ_INFO': False,               # Whether to use sequence information
    'SKIP_SPLIT_FOL': True,
    'DO_PREPROC': False,
}

metrics_config = {
    'METRICS': ['HOTA', 'CLEAR', 'Identity'],  # Which metrics to use (HOTA, CLEAR, Identity, VACE, etc.)
}

# Initialize the evaluator
evaluator = trackeval.Evaluator(eval_config)
dataset_list = [CustomMotDataset(dataset_config)]
metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]

# Run the evaluation
evaluator.evaluate(dataset_list, metrics_list)