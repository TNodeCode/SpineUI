import pandas as pd
from src.commands.tracking import StackTrackingCommand
from src.config.datasetconfig import DatasetConfiguration

selected_dataset = "spine_mixed_train"

# Get all available stacks in the dataset
stacks = DatasetConfiguration.get_dataset_stacks(dataset_name=selected_dataset)
stack_names = stacks.keys()
dfs = []

for stack_id, stack_name in enumerate(stack_names):
    cmd = StackTrackingCommand(dataset_name=selected_dataset, stack_name=stack_name)
    cmd.execute()
    cmd.traces_df['stack_id'] = stack_id
    dfs.append(cmd.traces_df)

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
df_traces.to_csv("tracking.csv", index=False)