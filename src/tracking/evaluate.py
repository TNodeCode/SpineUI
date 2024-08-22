import trackeval  # Import trackeval library
import multiprocessing


class CustomMotDataset(trackeval.datasets.MotChallenge2DBox):
    def __init__(self, config=None):
        super(CustomMotDataset, self).__init__(config=config)
        self.valid_class_numbers.extend([0])

    @staticmethod
    def evaluate(
        gt_folder: str,
        detections: str,
        output_dir: str = "tracking-results",
        num_parallel_cores: int = -1
    ):
        # Define paths and other configurations
        eval_config = {
            'USE_PARALLEL': False,          # Whether to use parallel processing
            'NUM_PARALLEL_CORES': num_parallel_cores if num_parallel_cores > 0 else multiprocessing.cpu_count(),        # Number of cores to use if parallel processing is enabled
            'PRINT_RESULTS': True,          # Whether to print the results
            'PRINT_CONFIG': True,           # Whether to print the configuration
            'TIME_PROGRESS': True,          # Whether to display the time progress
            'DISPLAY_LESS_PROGRESS': False, # Whether to display less progress
            'OUTPUT_SUMMARY': True,         # Whether to output a summary of the results
            'OUTPUT_DETAILED': True,        # Whether to output detailed results
            'PLOT_CURVES': True,           # Whether to plot curves
        }

        dataset_config = {
            'TRACKERS_FOLDER': detections,  # Folder containing the tracker results
            'GT_FOLDER': gt_folder,              # Folder containing the ground truth
            'SEQMAP_FILE': f"{gt_folder}/seqmaps.txt",          # File containing the sequence map
            'OUTPUT_FOLDER': output_dir,  # Where to save the outputs (if None, nothing is saved)
            'TRACKER_SUB_FOLDER': '',   # Subfolder within the tracker folder containing the data
            'BENCHMARK': 'MOT17',            # Which benchmark to use (MOT17, KITTI, etc.)
            'SEQ_INFO': False,               # Whether to use sequence information
            'SKIP_SPLIT_FOL': True,
            'DO_PREPROC': False,
        }

        # Initialize the evaluator
        evaluator = trackeval.Evaluator(eval_config)
        dataset_list = [CustomMotDataset(dataset_config)]
        metrics_list = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]

        # Run the evaluation
        evaluator.evaluate(dataset_list, metrics_list)