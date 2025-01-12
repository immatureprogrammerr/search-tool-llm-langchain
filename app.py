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

loadenv = load_dotenv()

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
                value="https://www.moneycontrol.com/news/business/markets/coronavirus-pandemic-markets-set-to-feel-the-heat-for-some-time-5026501.html",
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

content  = html.Div([
    html.Div(
        id="messages-div", 
        style={"border": "1px solid black", "padding": "10px", "marginTop": "10px"}),
    html.Div(
        id="page-content", 
        children=[html.H1("Search Tool 🔎")], 
        style=CONTENT_STYLE
    ),
    dcc.Store(id="process-step", data=0),
    dcc.Interval(
        id="process-interval", 
        interval=1000, 
        n_intervals=0, 
        disabled=True
    ),
])

app.layout = html.Div(
    [
        sidebar, 
        content
    ]
)

file_path = "faiss_file.pkl"
llm = OpenAI(temperature=0.9, max_tokens=500)

@app.callback(
    [
        Output('messages-div', 'children'),
        Output('page-content', 'children'),
        Output("process-step", "data")
    ],
    Input('submit-button', 'n_clicks'),
    [
        State('process-step', 'data'),
        State('messages-div', 'children')
    ],
    [State(f'input-box-{i}', 'value') for i in range(NUMBER_OF_INPUTS)]
)
def update_output(
    n_clicks, 
    current_step, 
    current_children,
    *values):
    if n_clicks > 0:
        if not isinstance(current_messages, list):
            current_messages = []

        # current_messages.append(html.Div(steps[current_step]))
        
        # load data
        loader = UnstructuredURLLoader(values)
        data = loader.load()

        current_children.append(
            html.Div('Text Splitter Started...⌛⌛⌛')
        )
        # split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        docs = text_splitter.split_documents(data)

        # create embeddings and save it to FAISS index
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        
        current_children.append(html.Div('Embedding Vector Started...⌛⌛⌛'))
        time.sleep(2)

        # save the FAISS index to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
        
        # ===== Added the place holder of `Question: `
        if query:
            if os.path.exists(file_path):
                with open(file_path, "rb") as f:
                    vectorstore = pickle.load(f)
                    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever)
    return current_children, values, current_step + 1

if __name__ == '__main__':
    app.run(debug=True)
