import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import pandas as pd
import os

dash.register_page(__name__, path='/training_evaluation')

# Set the path to the data directory
data_dir = "./training_metrics"

# Get the list of CSV files in the data directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Initialize the Dash app
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUMEN])

# Define the layout of the app
layout = html.Div([
    dbc.Container([
        dbc.Row([
            html.H3("Comparison of model training metrics for different training iterations"),
            html.P("Here you can see metrics like mAP50, recision and recall for the training process of different models.")
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='csv-dropdown-1-1',
                    options=[{'label': file, 'value': file} for file in csv_files],
                    value=[csv_files[0]],
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

# Define callback functions
@dash.callback(
    dash.dependencies.Output('line-chart-map50', 'figure'),
    dash.dependencies.Output('line-chart-precision', 'figure'),
    dash.dependencies.Output('line-chart-recall', 'figure'),
    dash.dependencies.Input('csv-dropdown-1-1', 'value'),
)
def update_line_charts(csv_files):
    dfs = []
    figure_data_ap = []
    figure_data_ar = []
    figure_data_f1 = []

    # Read the selected CSV files
    for i, csv_file in enumerate(csv_files):
        dfs.append(pd.read_csv(os.path.join(data_dir, csv_file)))
        figure_data_ap.append({
            'x': dfs[i].index,
            'y': dfs[i]['ap'],
            'name': csv_file
        })
        figure_data_ar.append({
            'x': dfs[i].index,
            'y': dfs[i]['ar'],
            'name': csv_file
        })
        figure_data_f1.append({
            'x': dfs[i].index,
            'y': dfs[i]['f1'],
            'name': csv_file
        })
    
    
    # Create line charts
    figure_map50 = {
        'data': figure_data_ap,
        'layout': {
            'title': f"AP scores during training",
            'xaxis': {'title': 'Epoch'},
            'yaxis': {'title': 'AP'},
        }
    }
    figure_precision = {
        'data': figure_data_ar,
        'layout': {
            'title': f"AR during training",
            'xaxis': {'title': 'Epoch'},
            'yaxis': {'title': 'AR'},
        }
    }
    figure_recall = {
        'data': figure_data_f1,
        'layout': {
            'title': f"F1 score during training",
            'xaxis': {'title': 'Epoch'},
            'yaxis': {'title': 'F1'},
        }
    }
    
    return figure_map50, figure_precision, figure_recall