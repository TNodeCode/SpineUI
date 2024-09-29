import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import os
import glob
import yaml

dash.register_page(__name__, path='/training_evaluation')

# Set the path to the data directory
root_directory = "./training_metrics"

# subdirectories
subdirs = os.listdir(root_directory)
subdir = subdirs[0]

# Get the list of CSV files in the data directory
csv_files = glob.glob(os.path.join(root_directory, subdir, "*.csv"))


figure_config = {
    'ap': {
        'title': 'AP scores during training',
        'x-axis': {'title': 'Epochs'},
        'y-axis': {'title': 'AP'},
    },
    'ar': {
        'title': 'AR scores during training',
        'x-axis': {'title': 'Epochs'},
        'y-axis': {'title': 'AR'},
    },
    'f1': {
        'title': 'F1 scores during training',
        'x-axis': {'title': 'Epochs'},
        'y-axis': {'title': 'F1'},
    },
}

yaml_file_path = root_directory + "/" + subdir + "/figures.yaml"

if os.path.exists(yaml_file_path):
    print("EXISTS")
    with open(yaml_file_path, "r") as fp:
        _figure_config = yaml.safe_load(fp)
        print(_figure_config)
        figure_config.update(_figure_config)
        print(figure_config)

# Define the layout of the app
layout = html.Div([
    dbc.Container([
        dbc.Row([
            html.P(),
            html.H3("Comparison of model training metrics for different training iterations"),
            html.P("Here you can see metrics like mAP50, recision and recall for the training process of different models.")
        ]),
        dbc.Row([
            dbc.Col(
                html.Label("Select subdirectory"),
                width=12
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='csv-dropdown-1-1',
                    options=[{'label': subdir, 'value': subdir} for subdir in os.listdir("./training_metrics")],
                    value=subdir,
                    multi=False,
                ),
                width=12
            ),
        ]),
        dbc.Row([
            dbc.Col([
                html.Br(),
                html.Label("Select Training Metrics"),
            ],
                width=12
            ),
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='csv-dropdown-2-1',
                    options=[{'label': file, 'value': file} for file in csv_files],
                    value=[],
                    multi=True,
                ),
                width=12
            ),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='line-chart-map50')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='line-chart-precision')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='line-chart-recall')
            ])
        ]),
    ])
])

@dash.callback(
    dash.dependencies.Output('csv-dropdown-2-1', 'options'),
    dash.dependencies.Output('csv-dropdown-2-1', 'value'),
    dash.dependencies.Input('csv-dropdown-1-1', 'value'),
)
def update_csv_dropdown(selected_subdirectory):
    # Get a list of CSV files in the selected subdirectory
    if selected_subdirectory:
        csv_files = [name for name in os.listdir(os.path.join(root_directory, selected_subdirectory)) if name.endswith('.csv')]
        return [{'label': name, 'value': name} for name in csv_files], []
    return [], None

# Define callback functions
@dash.callback(
    dash.dependencies.Output('line-chart-map50', 'figure'),
    dash.dependencies.Output('line-chart-precision', 'figure'),
    dash.dependencies.Output('line-chart-recall', 'figure'),
    dash.dependencies.Input('csv-dropdown-1-1', 'value'),
    dash.dependencies.Input('csv-dropdown-2-1', 'value'),
)
def update_line_charts(subdir, csv_files):
    if type(csv_files) is str:
        csv_files = [csv_files]
    dfs = []
    figure_data_ap = []
    figure_data_ar = []
    figure_data_f1 = []

    # Read the selected CSV files
    for i, csv_file in enumerate(csv_files):
        dfs.append(pd.read_csv(os.path.join(root_directory, subdir, csv_file)))
        figure_data_ap.append({
            'x': dfs[i].index,
            'y': dfs[i]['ap'],
            'name': os.path.basename(csv_file)
        })
        figure_data_ar.append({
            'x': dfs[i].index,
            'y': dfs[i]['ar'],
            'name': os.path.basename(csv_file)
        })
        figure_data_f1.append({
            'x': dfs[i].index,
            'y': dfs[i]['f1'],
            'name': os.path.basename(csv_file)
        })
    
    
    # Create line charts
    figure_map50 = {
        'data': figure_data_ap,
        'layout': {
            'title': figure_config['ap']['title'],
            'xaxis': figure_config['ap']['x-axis'],
            'yaxis': figure_config['ap']['y-axis'],
        }
    }
    figure_precision = {
        'data': figure_data_ar,
        'layout': {
            'title': figure_config['ar']['title'],
            'xaxis': figure_config['ar']['x-axis'],
            'yaxis': figure_config['ar']['y-axis'],
        }
    }
    figure_recall = {
        'data': figure_data_f1,
        'layout': {
            'title': figure_config['f1']['title'],
            'xaxis': figure_config['f1']['x-axis'],
            'yaxis': figure_config['f1']['y-axis'],
        }
    }
    
    return figure_map50, figure_precision, figure_recall