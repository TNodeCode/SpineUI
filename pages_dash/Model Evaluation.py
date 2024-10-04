import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import os

dash.register_page(__name__, path='/model_evaluation')

# Set the path to the data directory
data_dir = "./datasets/detections/faster_rcnn_mot"

# Get the list of CSV files in the data directory
csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

# Define the layout of the app
layout = html.Div([
    dbc.Container([
        dbc.Row([
            html.H3("Comparison of model predictions"),
            html.P("In this dashboard you can comare the predictions of multiple models.")
        ]),
        dbc.Row([
            dbc.Col(
                dcc.Dropdown(
                    id='bbox-csv-1',
                    options=[{'label': file, 'value': file} for file in csv_files],
                    value=csv_files[0],
                ),
                width=6
            ),
            dbc.Col(
                dcc.Dropdown(
                    id='bbox-csv-2',
                    options=[{'label': file, 'value': file} for file in csv_files],
                    value=csv_files[0],
                ),
                width=6
            ),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='bbox-dist-plot-1'), width=6),
            dbc.Col(dcc.Graph(id='bbox-dist-plot-2'), width=6),
        ])
    ])
])

@dash.callback(Output('bbox-dist-plot-1', 'figure'),
               Output('bbox-dist-plot-2', 'figure'),
               Input('bbox-csv-1', 'value'),
               Input('bbox-csv-2', 'value')
               )
def update_scatter_plot(csv_file_1, csv_file_2):
    print("CSV_FILE", csv_file_1, csv_file_2)
    
    df_1 = pd.read_csv(data_dir + "/" + csv_file_1)
    df_1["height"] = df_1["ymax"] - df_1["ymin"]
    df_1["width"] = df_1["xmax"] - df_1["xmin"]
    
    df_2 = pd.read_csv(data_dir + "/" + csv_file_2)
    df_2["height"] = df_2["ymax"] - df_2["ymin"]
    df_2["width"] = df_2["xmax"] - df_2["xmin"]

    x_min = 0
    x_max = 50
    y_min = 0
    y_max = 50
    
    # Create a scatter plot using Plotly Express
    #fig1 = px.scatter(df_1, x='width', y='height', hover_data=['class_name'])
    #fig2 = px.scatter(df_2, x='width', y='height', hover_data=['class_name'])
    # Create a 2D distribution plot using Plotly Figure Factory
    fig_dist_1 = ff.create_2d_density(
        df_1['width'],  # Width on x-axis
        df_1['height'],  # Height on y-axis
        colorscale='Blues',  # Set the colorscale
        hist_color='rgba(0, 0, 255, 0.5)'  # Set the histogram color
    )
    # Set the minimum and maximum limits for the x-axis and y-axis
    fig_dist_1.update_xaxes(range=[x_min, x_max])
    fig_dist_1.update_yaxes(range=[y_min, y_max])

    # Create a 2D distribution plot using Plotly Figure Factory
    fig_dist_2 = ff.create_2d_density(
        df_2['width'],  # Width on x-axis
        df_2['height'],  # Height on y-axis
        colorscale='Blues',  # Set the colorscale
        hist_color='rgba(0, 0, 255, 0.5)'  # Set the histogram color
    )
    # Set the minimum and maximum limits for the x-axis and y-axis
    fig_dist_2.update_xaxes(range=[x_min, x_max])
    fig_dist_2.update_yaxes(range=[y_min, y_max])

    
    return fig_dist_1, fig_dist_2