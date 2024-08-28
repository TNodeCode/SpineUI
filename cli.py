import os
import pandas as pd
from src.commands.tracking import StackTrackingCommand
from src.config.datasetconfig import DatasetConfiguration
from src.tracking.evaluate import CustomMotDataset
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--dataset', type=str, required=True, help='Dataset that should be used for object tracking')
@click.option('--detections', type=click.Path(exists=True, file_okay=True, dir_okay=False), required=True, help='Path to a CSV file containing detections')
@click.option('--output-dir', type=str, required=True, default='results_detection', help='Output directory')
def naive_tracking(dataset: str, detections: str, output_dir: str):
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
            }
            cmd.traces_df = cmd.traces_df.astype(convert_dict)
            cmd.traces_df.to_csv(f"{output_dir}/{stack_name}.txt", index=False, header=False)

        print("\r", end="")


    print(f"Saved tracking results at {output_dir}")


@cli.command()
@click.option('--gt-folder', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Path to MOT17 dataset')
@click.option('--detections', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Path to detections')
@click.option('--output-dir', type=str, required=False, help='Output directory')
def eval_tracking(gt_folder: str, detections: str, output_dir: str):
    CustomMotDataset.evaluate(
        gt_folder=gt_folder,
        detections=detections,
        output_dir=output_dir,
    )



if __name__ == '__main__':
    cli()