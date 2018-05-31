import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from plotly import tools

LOGFILE = 'examples/run_log.csv'

app = dash.Dash(__name__)
server = app.server


# Custom Script for Heroku
if 'DYNO' in os.environ:
    app.scripts.append_script({
        'external_url': 'https://cdn.rawgit.com/chriddyp/ca0d8f02a1659981a0ea7f013a378bbd/raw/e79f3f789517deec58f41251f7dbb6bee72c44ab/plotly_ga.js'
    })


def div_graph(name):
    """Generates an html Div containing graph and control options for smoothing and display, given the name"""
    return html.Div([
        html.Div(
            id=f'div-{name}-graph',
            className="ten columns"
        ),

        html.Div([
            html.Div([
                html.P("Smoothing:", style={'font-weight': 'bold', 'margin-bottom': '0px'}),

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
                html.P("Plot Display mode:", style={'font-weight': 'bold', 'margin-bottom': '0px'}),

                dcc.RadioItems(
                    options=[
                        {'label': 'Overlapping', 'value': 'overlap'},
                        {'label': 'Separate (Vertical)', 'value': 'separate_vertical'},
                        {'label': 'Separate (Horizontal)', 'value': 'separate_horizontal'}
                    ],
                    value='overlap',
                    id=f'radio-display-mode-{name}'
                ),

                html.Div(id=f'div-current-{name}-value')
            ]),
        ],
            className="two columns"
        ),
    ],
        className="row"
    )


app.layout = html.Div([
    # Banner display
    html.Div([
        html.H2(
            'Live Model Training Viewer',
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
        html.Div([
            html.Div([
                dcc.Dropdown(
                    id='dropdown-interval-control',
                    options=[
                        {'label': 'No Updates', 'value': 'no'},
                        {'label': 'Slow Updates', 'value': 'slow'},
                        {'label': 'Regular Updates', 'value': 'regular'},
                        {'label': 'Fast Updates', 'value': 'fast'}
                    ],
                    value='regular',
                    clearable=False,
                    searchable=False
                )
            ],
                className='ten columns'
            ),

            html.Div(
                id="div-step-display",
                className="two columns"
            )

        ],
            id='div-interval-control',
            className='row'
        ),

        dcc.Interval(
            id="interval-log-update",
            n_intervals=0
        ),

        # Hidden Div Storing JSON-serialized dataframe of run log
        html.Div(id='run-log-storage', style={'display': 'none'}),

        # The html divs storing the graphs and display parameters
        div_graph('accuracy'),
        div_graph('cross-entropy')
    ],
        className="container"
    )
])


def update_graph(graph_id,
                 graph_title,
                 y_train_index,
                 y_val_index,
                 run_log_json,
                 display_mode,
                 checklist_smoothing_options,
                 slider_smoothing):
    """
    :param graph_id: ID for Dash callbacks
    :param graph_title: Displayed on layout
    :param y_train_index: name of column index for y train we want to retrieve
    :param y_val_index: name of column index for y val we want to retrieve
    :param run_log_json: the json file containing the data
    :param display_mode: 'separate' or 'overlap'
    :param checklist_smoothing_options: 'train' or 'val'
    :param slider_smoothing: value between 0 and 1, at interval of 0.05
    :return: dcc Graph object containing the updated figures
    """
    def smooth(scalars, weight=0.6):
        last = scalars[0]
        smoothed = list()
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
        return smoothed

    if run_log_json:  # exists
        layout = go.Layout(
            title=graph_title,
            margin=go.Margin(l=50, r=50, b=50, t=50)
        )

        run_log_df = pd.read_json(run_log_json, orient='split')

        step = run_log_df['step']
        y_train = run_log_df[y_train_index]
        y_val = run_log_df[y_val_index]

        # Apply Smoothing if needed
        if 'train' in checklist_smoothing_options:
            y_train = smooth(y_train, weight=slider_smoothing)

        if 'val' in checklist_smoothing_options:
            y_val = smooth(y_val, weight=slider_smoothing)

        trace_train = go.Scatter(
            x=step,
            y=y_train,
            mode='lines',
            name='Training'
        )

        trace_val = go.Scatter(
            x=step,
            y=y_val,
            mode='lines',
            name='Validation'
        )

        if display_mode == 'separate_vertical':
            figure = tools.make_subplots(rows=2, cols=1, print_grid=False)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 2, 1)

            figure['layout'].update(title=layout.title,
                                    margin=layout.margin,
                                    height=layout.height)

        elif display_mode == 'separate_horizontal':
            figure = tools.make_subplots(rows=1, cols=2, print_grid=False)

            figure.append_trace(trace_train, 1, 1)
            figure.append_trace(trace_val, 1, 2)

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


@app.callback(Output('interval-log-update', 'interval'),
              [Input('dropdown-interval-control', 'value')])
def update_interval_log_update(interval_rate):
    if interval_rate == 'fast':
        return 500

    elif interval_rate == 'regular':
        return 1000

    elif interval_rate == 'slow':
        return 5 * 1000

    # Refreshes every 24 hours
    elif interval_rate == 'no':
        return 24 * 60 * 60 * 1000


@app.callback(Output('run-log-storage', 'children'),
              [Input('interval-log-update', 'n_intervals')])
def get_run_log(_):
    names = ['step', 'train accuracy', 'val accuracy', 'train cross entropy', 'val cross entropy']

    try:
        run_log_df = pd.read_csv(LOGFILE, names=names)
        json = run_log_df.to_json(orient='split')
    except FileNotFoundError as error:
        print(error)
        print("Please verify if the csv file generated by your model is placed in the correct directory.")
        return None

    return json


@app.callback(Output('div-step-display', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_step_display(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return html.H6(f"Step: {run_log_df['step'].iloc[-1]}", style={'margin-top': '3px'})


@app.callback(Output('div-accuracy-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('radio-display-mode-accuracy', 'value'),
               Input('checklist-smoothing-options-accuracy', 'values'),
               Input('slider-smoothing-accuracy', 'value')])
def update_accuracy_graph(run_log_json,
                          display_mode,
                          checklist_smoothing_options,
                          slider_smoothing):
    figure = update_graph('accuracy-graph',
                          'Prediction Accuracy',
                          'train accuracy',
                          'val accuracy',
                          run_log_json,
                          display_mode,
                          checklist_smoothing_options,
                          slider_smoothing)
    return [figure]


@app.callback(Output('div-cross-entropy-graph', 'children'),
              [Input('run-log-storage', 'children'),
               Input('radio-display-mode-cross-entropy', 'value'),
               Input('checklist-smoothing-options-cross-entropy', 'values'),
               Input('slider-smoothing-cross-entropy', 'value')])
def update_cross_entropy_graph(run_log_json,
                               display_mode,
                               checklist_smoothing_options,
                               slider_smoothing):
    figure = update_graph('cross-entropy-graph',
                          'Cross Entropy Loss',
                          'train cross entropy',
                          'val cross entropy',
                          run_log_json,
                          display_mode,
                          checklist_smoothing_options,
                          slider_smoothing)
    return [figure]


@app.callback(Output('div-current-accuracy-value', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_current_accuracy_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return [
            html.P(
                "Current Accuracy:",
                style={
                    'font-weight': 'bold',
                    'margin-top': '15px',
                    'margin-bottom': '0px'
                }
            ),
            html.Div(f"Training: {run_log_df['train accuracy'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val accuracy'].iloc[-1]:.4f}")
        ]


@app.callback(Output('div-current-cross-entropy-value', 'children'),
              [Input('run-log-storage', 'children')])
def update_div_current_cross_entropy_value(run_log_json):
    if run_log_json:
        run_log_df = pd.read_json(run_log_json, orient='split')
        return [
            html.P(
                "Current Loss:",
                style={
                    'font-weight': 'bold',
                    'margin-top': '15px',
                    'margin-bottom': '0px'
                }
            ),
            html.Div(f"Training: {run_log_df['train cross entropy'].iloc[-1]:.4f}"),
            html.Div(f"Validation: {run_log_df['val cross entropy'].iloc[-1]:.4f}")
        ]


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
