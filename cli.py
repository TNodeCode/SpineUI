import os
import numpy as np
import pandas as pd
import shutil
from src.commands.tracking import StackTrackingCommand
from src.config.datasetconfig import DatasetConfiguration
from src.tracking.evaluate import CustomMotDataset
from src.bytetrack.bytetrack import track as _bytetrack
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--dataset', type=str, required=True, help='Dataset that should be used for object tracking')
@click.option('--detections', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Path to a CSV file containing detections')
@click.option('--output-dir', type=str, required=True, default='results_detection', help='Output directory')
@click.option('--threshold', type=float, required=False, default=0.5, help='minimum confidence score')
def naive_tracking(dataset: str, detections: str, output_dir: str, threshold: float):
    print("Perform tracking ...")
    # Get all available stacks in the dataset
    stacks = DatasetConfiguration.get_dataset_stacks(dataset_name=dataset)
    stack_names = stacks.keys()
    output_dir = f"{output_dir}/seqmaps/"
    os.makedirs(output_dir, exist_ok=True)

    for stack_id, stack_name in enumerate(stack_names):
        print(f"Stack {stack_id+1}/{len(stack_names)} ({stack_name})", end="")
        cmd = StackTrackingCommand(
            dataset_name=dataset,
            stack_name=stack_name,
            detections_file=detections
        )
        cmd.execute()
        
        if cmd.traces_df.shape[0] > 0:
            # Remove columns that are not needed by TrackEval
            cmd.traces_df = cmd.traces_df.drop(columns=['filename', 'basename','cx','cy'])
            # Add x, y and z columns
            cmd.traces_df['x'] = -1
            cmd.traces_df['y'] = -1
            cmd.traces_df['z'] = -1
            convert_dict = {
                'frame': int,
                'object_id': int,
                'x0': int,
                'y0': int,
                'w': int,
                'h': int,
                'score': float,
            }
            cmd.traces_df = cmd.traces_df.astype(convert_dict)
            cmd.traces_df = cmd.traces_df[cmd.traces_df['score'] >= threshold]
            cmd.traces_df.to_csv(f"{output_dir}/{stack_name}.txt", index=False, header=False)

        print("\r", end="")


    print(f"Saved tracking results at {output_dir}")


@cli.command()
@click.option('--detections', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Path to a CSV file containing detections')
@click.option('--output-dir', type=str, required=True, default='results_detection', help='Output directory')
@click.option('--track_threshold', type=float, required=False, default=0.5, help='track threshold')
@click.option('--match_threshold', type=float, required=False, default=0.7, help='match threshold')
@click.option('--track_buffer', type=int, required=False, default=100, help='track buffer')
def bytetrack(detections: str, output_dir: str, track_threshold: float, match_threshold: float, track_buffer: int):
    # Initialize ByteTrack
    class Args:
        match_thresh = 0.5
        track_thresh = 0.3
        track_buffer = 100
        mot20 = False
        use_cuda = False  # Set to True if you're using a GPU

    # Initialize args
    args = Args()
    args.track_thresh = track_threshold
    args.match_thresh = match_threshold
    args.track_buffer = track_buffer

    # Load the CSV detection data
    df = pd.read_csv(detections, header=None)

    # Rename columns
    df.columns = ['frame', 'class_index', 'class_name', 'x0', 'y0', 'x1', 'y1', 'score']

    # Get unique stack names
    stack_names = pd.Series(df['frame'].apply(lambda x: os.path.dirname(x)).unique())
    stack_names = stack_names.apply(lambda x: x.split('/')[0])

    for i, stack_name in enumerate(stack_names):
        if not stack_name: # skip empty stack names
            continue
        print(f"Processing stack {i+1}/{len(stack_names)} {stack_name} ...")

        # Filter df by stack name
        df_stack = df[df['frame'].str.contains(stack_name)]
        df_stack.loc[:, 'x0'] = df_stack.loc[:, 'x0'].astype(int)
        df_stack.loc[:, 'x1'] = df_stack.loc[:, 'x1'].astype(int)
        df_stack.loc[:, 'y0'] = df_stack.loc[:, 'y0'].astype(int)
        df_stack.loc[:, 'y1'] = df_stack.loc[:, 'y1'].astype(int)
        df_stack.loc[:, 'score'] = df_stack.loc[:, 'score'].astype(float)
        df_stack = df_stack[df_stack['score'] > 0.5]

        # Perform tracking with ByteTrack
        _bytetrack(args, df_stack, output_dir=output_dir, stack_name=stack_name)

    print("Tracking complete!")


@cli.command()
@click.option('--src-dir', type=str, required=True, help='Input directory')
@click.option('--dst-dir', type=str, required=True, help='Output directory')
def generate_seqmaps(
        src_dir: str,
        dst_dir: str,
):
    print(f"Copy results from {src_dir} to {dst_dir} ...")
    os.makedirs(dst_dir, exist_ok=True)

    for subdir in os.listdir(src_dir):
        path_src = f"{src_dir}/{subdir}/seqmaps"
        filenames = os.listdir(path_src)
        path_dst = dst_dir + "/" + subdir
        os.makedirs(path_dst, exist_ok=True)

        for filename in filenames:
            shutil.copy2(
                src=path_src + "/" + filename,
                dst=path_dst + "/" + filename.replace(".csv", ".txt")
            )
    print("Finished preprocessing")



@cli.command()
@click.option('--gt-folder', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Path to MOT17 dataset')
@click.option('--detections', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Path to detections')
@click.option('--output-dir', type=str, required=False, help='Output directory')
@click.option('--similarity-metric', type=str, required=False, default='IoU', help='Metric used for computing similarities of bboxes (IoU or IoM)')
def eval_tracking(gt_folder: str, detections: str, output_dir: str, similarity_metric: str):
    CustomMotDataset.evaluate(
        gt_folder=gt_folder,
        detections=detections,
        output_dir=output_dir,
        metric=similarity_metric,
    )


@cli.command()
@click.option('--src-dir', type=str, required=True, help='Input directory')
def summarize_tracking(src_dir: str):
    epoch_dfs = []
    epoch_dfs_detailed = []

    for sub_dir in os.listdir(src_dir):
        if os.path.isfile(f"{src_dir}/{sub_dir}"):
            continue
        df_epoch = pd.read_csv(f"{src_dir}/{sub_dir}/spine_summary.txt", sep=" ")
        df_detailed = pd.read_csv(f"{src_dir}/{sub_dir}/spine_detailed.csv")
        epoch = int(sub_dir.replace("epoch_", ""))
        df_epoch['epoch'] = [epoch]
        df_detailed['epoch'] = np.ones(df_detailed.shape[0]) * epoch
        epoch_dfs.append(df_epoch)
        epoch_dfs_detailed.append(df_detailed)

    df_summary = pd.concat(epoch_dfs, ignore_index=True)
    df_summary_detailed = pd.concat(epoch_dfs_detailed, ignore_index=True)

    df_summary.to_csv(f"{src_dir}/summary.csv")
    df_summary_detailed.to_csv(f"{src_dir}/summary_detailed.csv")




if __name__ == '__main__':
    cli()