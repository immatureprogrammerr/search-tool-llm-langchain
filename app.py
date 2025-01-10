import os
import pickle
import time
from dash import Dash, html, dcc, State
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

NUMBER_OF_INPUTS = 3

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "12em",
    "padding": "2rem 1rem",
    "backgroundColor": "#2b2b2b",
    "color": "#cfcfcf",
    "fontSize": "23px",
    "boxShadow": "5px 5px 5px 5px lightgrey"
}

CONTENT_STYLE = {
    "marginLeft": "18rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem"
}

sidebar = html.Div(
    [
        html.H1(f"Search Tool", style={'fontSize': '36px', 'fontWeight': 'bold'}),
        html.Hr(),
        html.H2(f"Enter articles URLs", className="lead", style={'fontSize': '28px'}),
        html.Hr(),
        dbc.Nav([
            html.Div([dcc.Input(
                id=f"input-box-{i}",
                type="text",
                placeholder=f"Enter URL {i + 1}",
                style={'marginTop': '10px'}
            ) for i in range(NUMBER_OF_INPUTS)]),
            html.Button('Submit', id='submit-button', n_clicks=0, style={'width': '50%', 'margin': '20px auto'})
        ],
        vertical=True,
        pills=True,
        )
    ],
    style=SIDEBAR_STYLE
)

app =  Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

content  = html.Div(id="page-content", style=CONTENT_STYLE)
app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

@app.callback(
    Output('page-content', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State(f'input-box-{i}', 'value') for i in range(NUMBER_OF_INPUTS)]
)
def update_output(n_clicks, *values):
    if n_clicks > 0:
        return f"You entered: {values}"
    return "Waiting for input ..."

if __name__ == '__main__':
    app.run(debug=True)
