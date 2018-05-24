import time
from scipy.interpolate import interp1d
from scipy.interpolate import spline
import plotly.graph_objs as go
import csv
import os
import pandas as pd
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)
server = app.server

# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })


app.layout = html.Div([
    # Banner display
    html.Div([
        html.H2(
            'App Name',
            id='title'
        ),
        html.Img(
            src="https://s3-us-west-1.amazonaws.com/plotly-tutorials/logo/new-branding/dash-logo-by-plotly-stripe-inverted.png"
        )
    ],
        className="banner"
    ),

    # Body
    html.Div([
        dcc.Interval(
            id="interval-component",
            interval=500,
            n_intervals=0
        ),

        html.Div(id='run-log-storage', style={'display': 'none'}),

        dcc.Graph(
            id="accuracy-graph"
        ),




    ],
        className="container",
    )

])


def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

# @app.server.before_first_request
# def load_csv():
#     global csv_reader, step, train_error, val_error
#
#     csvfile = open('run_log.csv', 'r', newline='')
#     csv_reader = csv.reader(csvfile, delimiter=',')
#     step = []
#     train_error = []
#     val_error = []


@app.callback(Output('run-log-storage', 'children'),
              [Input('interval-component', 'n_intervals')])
def get_run_log(n_intervals):
    t1 = time.time()

    names = ['step', 'train accuracy', 'val accuracy', 'train cross entropy', 'val cross entropy']
    run_log_df = pd.read_csv('run_log.csv', names=names)
    json = run_log_df.to_json(orient='split')

    t2 = time.time()
    print(f"\ncsv2json time: {t2-t1:.3f} sec")

    return json


@app.callback(Output('accuracy-graph', 'figure'),
              [Input('run-log-storage', 'children')])
def update_accuracy_curve(run_log_json):
    t1 = time.time()
    run_log_df = pd.read_json(run_log_json, orient='split')
    t2 = time.time()
    print(f"json2csv time: {t2-t1:.3f} sec\n")

    layout = go.Layout(
        title="Prediction Accuracy",
        margin=go.Margin(l=50, r=50, b=50, t=50)
    )

    # for row in csv_reader:
    #     step.append(int(row[0]))
    #     train_error.append(float(row[1]))
    #     val_error.append(float(row[2]))

    step = run_log_df['step']
    train_accuracy = run_log_df['train accuracy']
    val_accuracy = run_log_df['val accuracy']

    trace_train = go.Scatter(
        x=step,
        y=train_accuracy,
        mode='lines',
        name='Training'
    )

    trace_val = go.Scatter(
        x=step,
        y=smooth(val_accuracy),
        mode='lines',
        name='Validation'
    )

    return go.Figure(
        data=[trace_train, trace_val],
        layout=layout
    )

    # return go.Figure(go.Scatter(visible=False))


external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",  # Normalize the CSS
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto"  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    "https://rawgit.com/xhlulu/0acba79000a3fd1e6f552ed82edb8a64/raw/dash_template.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)