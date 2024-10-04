import dash
from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import os
import glob
import yaml

dash.register_page(__name__, path='/mot_evaluation')

# Set the path to the data directory
root_directory = "./datasets/mot_metrics"

# subdirectories
subdirs = os.listdir(root_directory)
subdir = subdirs[0]

# Get the list of CSV files in the data directory
csv_files = glob.glob(os.path.join(root_directory, subdir, "*.csv"))

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
                    id='mot-csv-dropdown-1-1',
                    options=[{'label': subdir, 'value': subdir} for subdir in os.listdir(root_directory)],
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
                    id='mot-csv-dropdown-2-1',
                    options=[{'label': file, 'value': file} for file in csv_files],
                    value=[],
                    multi=False,
                ),
                width=12
            ),
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='mot-hota-scores')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='mot-clearmot-scores')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H3("Ground Truth and Detections")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id='mot-row-datatable-1')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H3("MOTA Metrics")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id='mot-row-datatable-2')
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.H3("Identity Metrics")
            ])
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(id='mot-row-datatable-3')
            ])
        ]),
    ])
])

@dash.callback(
    dash.dependencies.Output('mot-csv-dropdown-2-1', 'options'),
    dash.dependencies.Output('mot-csv-dropdown-2-1', 'value'),
    dash.dependencies.Input('mot-csv-dropdown-1-1', 'value'),
)
def update_csv_dropdown(selected_subdirectory):
    # Get a list of CSV files in the selected subdirectory
    if selected_subdirectory:
        csv_files = [name for name in os.listdir(os.path.join(root_directory, selected_subdirectory)) if name.endswith('.csv')]
        return [{'label': name, 'value': name} for name in csv_files], []
    return [], None

# Define callback functions
@dash.callback(
    dash.dependencies.Output('mot-hota-scores', 'figure'),
    dash.dependencies.Output('mot-clearmot-scores', 'figure'),
    dash.dependencies.Output('mot-row-datatable-1', 'children'),
    dash.dependencies.Output('mot-row-datatable-2', 'children'),
    dash.dependencies.Output('mot-row-datatable-3', 'children'),
    dash.dependencies.Input('mot-csv-dropdown-1-1', 'value'),
    dash.dependencies.Input('mot-csv-dropdown-2-1', 'value'),
)
def update_line_charts(subdir, csv_file):
    figure_data_hota = []
    figure_data_clearmot = []
    dt_1, dt_2, dt_3 = None, None, None

    figure_config = {
        'hota': {
            'title': 'MOT metrics',
            'x-axis': {'title': 'Alpha'},
            'y-axis': {'title': 'Score'},
        },
        'hota_tpfpfn': {
            'title': 'MOT TP, FP, FN',
            'x-axis': {'title': 'Alpha'},
            'y-axis': {'title': 'Number'},
        },
    }

    yaml_file_path = root_directory + "/" + subdir + "/figures.yaml"

    if os.path.exists(yaml_file_path):
        with open(yaml_file_path, "r") as fp:
            _figure_config = yaml.safe_load(fp)
            figure_config.update(_figure_config)

    # Read the selected CSV files
    if csv_file:
        df = pd.read_csv(os.path.join(root_directory, subdir, csv_file), index_col='seq')
        metrics = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr', 'LocA']
        styles = [
            dict(color='red'),
            dict(color='blue'),
            dict(color='green'),
            dict(dash='dash', color='blue'),
            dict(dash='dot', color='blue'),
            dict(dash='dash', color='green'),
            dict(dash='dot', color='green'),
            dict(color='purple')
        ]
        for metric, style in zip(metrics, styles):
            figure_data_hota.append({
                'x': [x/100 for x in range(5, 100, 5)],
                'y': [df.loc['COMBINED'][f"{metric}___{alpha}"] for alpha in range(5, 100, 5)],
                'name': metric +  " (" + str(round(df.loc['COMBINED'][f"{metric}___AUC"], 2)) + ")",
                'line': style
            }) 

        styles = [ 
            dict(color='green'),
            dict(color='orange'),
            dict(color='red'),
        ]

        for metric, name, style in zip(['HOTA_TP', 'HOTA_FP', 'HOTA_FN'], ['TP', 'FP', 'FN'], styles):
            figure_data_clearmot.append({
                'x': [x/100 for x in range(5, 100, 5)],
                'y': [df.loc['COMBINED'][f"{metric}___{alpha}"] for alpha in range(5, 100, 5)],
                'name': name,
                'line': style
            })

        table_columns_1 = ['MOTA','MOTP','MODA']
        table_columns_2 = ['IDF1','IDR','IDP','IDTP','IDFN','IDFP']
        table_columns_3 = ['Dets','GT_Dets','IDs','GT_IDs']
        df_table_1 = df.loc[:, table_columns_1]
        df_table_2 = df.loc[:, table_columns_2]
        df_table_3 = df.loc[:, table_columns_3]

        dt_1 = dash_table.DataTable(
            data=df_table_1[-1:].to_dict('records'),
            columns=[{'name': str(i), 'id': str(i)} for i in df_table_1.columns],
            id='df-metrics-1'
        )

        dt_2 = dash_table.DataTable(
            data=df_table_2[-1:].to_dict('records'),
            columns=[{'name': str(i), 'id': str(i)} for i in df_table_2.columns],
            id='df-metrics-2'
        )

        dt_3 = dash_table.DataTable(
            data=df_table_3[-1:].to_dict('records'),
            columns=[{'name': str(i), 'id': str(i)} for i in df_table_3.columns],
            id='df-metrics-3'
        )
    
    # Create line charts
    figure_hota = {
        'data': figure_data_hota,
        'layout': {
            'title': figure_config['hota']['title'],
            'xaxis': figure_config['hota']['x-axis'],
            'yaxis': figure_config['hota']['y-axis'],
        }
    } 
    
    # Create line charts
    figure_clearmot = {
        'data': figure_data_clearmot,
        'layout': {
            'title': figure_config['hota_tpfpfn']['title'],
            'xaxis': figure_config['hota_tpfpfn']['x-axis'],
            'yaxis': figure_config['hota_tpfpfn']['y-axis'],
        }
    }
    
    return figure_hota, figure_clearmot, dt_1, dt_2, dt_3