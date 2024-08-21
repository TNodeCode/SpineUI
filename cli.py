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
@click.option('--output', type=str, required=True, default='tracking.csv', help='Output file')
def naive_tracking(dataset, detections, output):
    print("Perform tracking ...")
    # Get all available stacks in the dataset
    stacks = DatasetConfiguration.get_dataset_stacks(dataset_name=dataset)
    stack_names = stacks.keys()
    dfs = []

    for stack_id, stack_name in enumerate(stack_names):
        print(f"Stack {stack_id+1}/{len(stack_names)} ({stack_name})", end="")
        cmd = StackTrackingCommand(
            dataset_name=dataset,
            stack_name=stack_name,
            detections_file=detections
        )
        cmd.execute()
        cmd.traces_df['stack_id'] = stack_id
        dfs.append(cmd.traces_df)
        print("\r", end="")

    df_traces = pd.concat(dfs)

    if df_traces.shape[0] > 0:
        convert_dict = {
            'object_id': int,
            'frame': int,
            'cx': int,
            'cy': int,
            'w': int,
            'h': int,
        }
    df_traces = df_traces.astype(convert_dict)
    df_traces.to_csv(output, index=False)

    print(f"Saved tracking results at {output}")


@cli.command()
def eval_tracking():
    CustomMotDataset.evaluate()



if __name__ == '__main__':
    cli()