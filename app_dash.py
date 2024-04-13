import dash
from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, use_pages=True, pages_folder="./pages_dash", external_stylesheets=[dbc.themes.LUMEN])

app.layout = html.Div([
    dbc.NavbarSimple(
        children=[
            dbc.DropdownMenu(
                children=[
                    dbc.DropdownMenuItem("Pages", header=True)
                ] + [
                    dbc.DropdownMenuItem(page['name'], href=page['path']) for page in dash.page_registry.values()
                ],
                nav=True,
                in_navbar=True,
                label="Pages",
            ),
        ],
        brand="Detection Model Analysis",
        brand_href="#",
        color="primary",
        dark=True,
    ),
    dash.page_container
])

if __name__ == '__main__':
    app.run(debug=True)