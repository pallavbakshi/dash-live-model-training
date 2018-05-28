import time
from scipy.interpolate import interp1d
from scipy.interpolate import spline
import plotly.graph_objs as go
import csv
import os
import pandas as pd
from plotly import tools
import numpy as np
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html

app = dash.Dash(__name__)
server = app.server


def div_graph(name):
    return html.Div([
        html.Div(
            id=f'div-{name}-graph',
            className="ten columns"
        ),

        html.Div([
            html.Div([
                "Smoothing:",

                dcc.Checklist(
                    options=[
                        {'label': 'Training', 'value': 'train'},
                        {'label': 'Validation', 'value': 'val'}
                    ],
                    values=[],
                    id=f'checklist-smoothing-options-{name}'
                )
            ],
                style={'margin-top': '10px'}
            ),

            html.Div([
                dcc.Slider(
                    min=0,
                    max=1,
                    step=0.05,
                    marks={i / 5: i / 5 for i in range(0, 6)},
                    value=0.6,
                    id=f'slider-smoothing-{name}'
                )
            ],
                style={'margin-bottom': '40px'}
            ),

            html.Div([
                "Display mode:",

                dcc.RadioItems(
                    options=[
                        {'label': 'Overlapping Plots', 'value': 'overlap'},
                        {'label': 'Separate Plots', 'value': 'separate'}
                    ],
                    value='overlap',
                    id=f'radio-display-mode-{name}'
                )
            ]),
        ],
            className="two columns"
        ),
    ],
        className="row"
    )

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
            id="interval-log-update",
            interval=50000,
            n_intervals=0
        ),

        html.Div(id='run-log-storage', style={'display': 'none'}),

        div_graph('accuracy'),

        dcc.Graph(id="cross-entropy-graph"),

    ],
        className="container"
    )
])


def smooth(scalars, weight=0.6):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def update_graph(graph_id,
                 graph_title,
                 run_log_json,
                 display_mode,
                 checklist_smoothing_options,
                 slider_smoothing):
    if run_log_json:  # exists
        layout = go.Layout(
            title=graph_title,
            margin=go.Margin(l=50, r=50, b=50, t=50)
        )

        t1 = time.time()
        run_log_df = pd.read_json(run_log_json, orient='split')
        t2 = time.time()
        print(f"json2csv time: {t2-t1:.3f} sec\n")

        step = run_log_df['step']
        train_accuracy = run_log_df['train accuracy']
        val_accuracy = run_log_df['val accuracy']

        # Apply Smoothing if needed
        if 'train' in checklist_smoothing_options:
            train_accuracy = smooth(train_accuracy, weight=slider_smoothing)

        if 'val' in checklist_smoothing_options:
            val_accuracy = smooth(val_accuracy, weight=slider_smoothing)

        trace_train = go.Scatter(
            x=step,
            y=train_accuracy,
            mode='lines',
            name='Training'
        )

        trace_val = go.Scatter(
            x=step,
            y=val_accuracy,
            mode='lines',
            name='Validation'
        )

        if display_mode == 'separate':
            figure = tools.make_subplots(rows=2, cols=1)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 2, 1)

            figure['layout'].update(title=layout.title,
                                    margin=layout.margin,
                                    height=layout.height)

        elif display_mode == 'overlap':
            figure = go.Figure(
                data=[trace_train, trace_val],
                layout=layout
            )

        else:
            figure = None

        return dcc.Graph(figure=figure, id=graph_id)

    return dcc.Graph(id=graph_id)


@app.callback(Output('run-log-storage', 'children'),
              [Input('interval-log-update', 'n_intervals')])
def get_run_log(n_intervals):
    t1 = time.time()

    names = ['step', 'train accuracy', 'val accuracy', 'train cross entropy', 'val cross entropy']

    try:
        run_log_df = pd.read_csv('run_log.csv', names=names)
        json = run_log_df.to_json(orient='split')
    except FileNotFoundError as error:
        print(error + ". Please verify if the csv file generated by your model is place in the correct directory.")
        return None

    t2 = time.time()
    print(f"\ncsv2json time: {t2-t1:.3f} sec")

    return json


@app.callback(Output('div-accuracy-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('radio-display-mode-accuracy', 'value'),
               Input('checklist-smoothing-options-accuracy', 'values'),
               Input('slider-smoothing-accuracy', 'value')])
def update_accuracy_graph(run_log_json, display_mode, checklist_smoothing_options, slider_smoothing):
    figure = update_graph('accuracy-graph',
                          'Prediction Accuracy',
                          run_log_json,
                          display_mode,
                          checklist_smoothing_options,
                          slider_smoothing)

    return [figure]


@app.callback(Output('cross-entropy-graph', 'figure'),
              [Input('run-log-storage', 'children')])
def update_cross_entropy_graph(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')

        layout = go.Layout(
            title="Cross Entropy Loss",
            margin=go.Margin(l=50, r=50, b=50, t=50)
        )

        step = run_log_df['step']
        train_accuracy = run_log_df['train cross entropy']
        val_accuracy = run_log_df['val cross entropy']

        trace_train = go.Scatter(
            x=step,
            y=train_accuracy,
            mode='lines',
            name='Training'
        )

        trace_val = go.Scatter(
            x=step,
            y=val_accuracy,
            mode='lines',
            name='Validation'
        )

        return go.Figure(
            data=[trace_train, trace_val],
            layout=layout
        )

    return go.Figure(go.Scatter(visible=False))


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
