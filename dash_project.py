import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_daq as daq
import math
from clusters import plot_clusters

# data from https://github.com/owid/covid-19-data/tree/master/public/data
# link to data: https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv
df = pd.read_csv('https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv')    # replace with link if you want to see the updated data (csv-file contains data up to 13.09.2020)
continents = ['Europe', 'Asia', 'North America', 'South America', 'Oceania', 'Africa']
vars = ['total_cases', 'population', 'median_age']
df = df[df['continent'].isin(continents)]
dates = pd.to_datetime(df['date']).dt.date.unique()
locations = df.location.unique()
indicators = ['total_cases', 'new_cases', 'new_cases_smoothed', 'total_deaths', 'new_deaths', 'new_deaths_smoothed',
 'total_cases_per_million', 'new_cases_per_million', 'new_cases_smoothed_per_million', 'total_deaths_per_million',
 'new_deaths_per_million', 'new_deaths_smoothed_per_million', 'total_tests', 'new_tests', 'new_tests_smoothed',
 'total_tests_per_thousand', 'new_tests_per_thousand', 'new_tests_smoothed_per_thousand', 'tests_per_case',
 'positive_rate', 'tests_units', 'stringency_index', 'population', 'population_density', 'median_age', 'aged_65_older',
 'aged_70_older', 'gdp_per_capita', 'extreme_poverty', 'cardiovasc_death_rate', 'diabetes_prevalence', 'female_smokers',
 'male_smokers', 'handwashing_facilities', 'hospital_beds_per_thousand', 'life_expectancy', 'human_development_index']

indicators_ref = [k.replace('_', ' ') for k in indicators]
indicators_dict = dict(zip(indicators_ref, indicators))
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']


theme =  {
    'dark': True,
    'detail': '#007439',
    'primary': '#111111',
    'secondary': '#6E6E6E',
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


rootLayout = html.Div(children=[
    html.Div([
        html.H1(
            children='COVID-19 dashboard: visualizing and clustering SARS-CoV-2 data',
            style={
                'textAlign': 'center',
                'color': 'yellow'
            }
        )
    ]),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i} for i in indicators_ref],
                value='new cases',
                style={'color': 'black'},
                clearable=False
            ),
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i} for i in indicators_ref],
                value='new deaths',
                style={'color': 'black'},
                clearable=False
            ),

        ],
        style={'width': '10%', 'display': 'inline-block'}),

        html.Div([

            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Log',
                labelStyle={'display': 'inline-block'},
                style={'color': 'white'}
            ),

            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Log',
                labelStyle={'display': 'inline-block', 'marginTop': '19px'},
                style={'color': 'white'}
            )
        ],
            style={'width': '10%', 'display': 'inline-block', 'margin-left': '15px'}),


        html.Div([
            dcc.DatePickerSingle(
                id='date-picker-single',
                min_date_allowed=dates.min(),
                max_date_allowed=dates.max(),
                initial_visible_month=dates.min(),
                date='2020-08-31',
                style={'display': 'inline-block', 'margin-left': '10px'}
            )
        ],
            style={'display': 'inline-block'}),

        html.Div([
            html.P('Country', style={'color': 'white'}),
            dcc.Dropdown(
                id='countries',
                options=[{'label': i, 'value': i} for i in locations],
                value=locations[0],
                style={'width': '100%', 'color': 'black', 'display': 'inline-block'},
                clearable=False
            ),
        ],
            style={'width': '15%', 'display': 'inline-block', 'margin-left': '200px'}),

        html.Div([
            html.P('Region 1', style={'color': 'white'}),
            dcc.Dropdown(
                id='continent-1',
                options=[{'label': i, 'value': i} for i in continents],
                value=continents[0],
                style={'width': '80%', 'color': 'black', 'display': 'inline-block'},
                clearable=False
            ),
        ],
            style={'width': '10%', 'display': 'inline-block', 'margin-left': '100px'}),

        html.Div([
            html.P('Region 2', style={'color': 'white'}),
            dcc.Dropdown(
                id='continent-2',
                options=[{'label': i, 'value': i} for i in continents],
                value=continents[1],
                style={'width': '80%', 'color': 'black', 'display': 'inline-block'},
                clearable=False
            ),
        ],
            style={'width': '10%', 'display': 'inline-block'}),

        html.Div([
            html.P('Region 3', style={'color': 'white'}),
            dcc.Dropdown(
                id='continent-3',
                options=[{'label': i, 'value': i} for i in continents],
                value=continents[2],
                style={'width': '80%', 'color': 'black', 'display': 'inline-block'},
                clearable=False
            ),
        ],
            style={'width': '10%', 'display': 'inline-block'}),

        dcc.Checklist(
            id='checklist',
            options=[
                {'label': 'All regions', 'value': 'All'},
            ],
            value=['All'],
            style={'display': 'inline-block', 'color': 'white'}
        ),

    ],
    style={
        'backgroundColor': '##111111',
        'padding': '10px 5px',
    }),

    html.Div([
        html.Div([
            dcc.Graph(
                id='crossfilter-indicator-scatter',
                hoverData={'points': [{'customdata': 'Russia'}]}
            )
        ], style={'display': 'inline-block', 'padding': '0 20', 'width': '33%'}),


        html.Div([
            dcc.Graph(id='x-time-series'),
            dcc.Graph(id='y-time-series'),
        ], style={'display': 'inline-block', 'float': 'center', 'padding': '0 20', 'margin-right': '40px'}),

        html.Div([
            html.Div([
                dcc.Graph(id='top5-countries')
            ]),
            html.Div([
                html.P('New tests per 1K', style={'color': 'white'}),
                daq.LEDDisplay(
                    id='new-tests-led',
                    value='1704',
                    color='#92e0d3',
                    size=50,
                    style={'margin-bottom': '20px'}
                ),

                html.P('New cases per 1M', style={'color': 'white'}),
                daq.LEDDisplay(
                    id='new-cases-led',
                    value='1704',
                    color='#92e0d3',
                    size=50,
                ),
            ], style={'display': 'inline-block', 'width': '33%'}),

            html.Div([
                html.P('New deaths per 1M', style={'color': 'white'}),
                daq.LEDDisplay(
                    id='new-deaths-led',
                    value='1704',
                    color='#92e0d3',
                    size=50,
                    style={'margin-bottom': '20px'}
                ),

                html.P('Positive rate', style={'color': 'white'}),
                daq.LEDDisplay(
                    id='positive-rate',
                    value='1704',
                    color='#92e0d3',
                    size=50,
                ),
            ], style={'display': 'inline-block', 'margin-left': '70px'}),
        ], style={'display': 'inline-block'})
    ], style={'borderBottom': 'thin lightgrey solid'}),

    html.Div([
        html.Div([
            dcc.Graph(
                id='3d-indicators-scatter-actual',
            )
        ], style={'display': 'inline-block', 'padding': '0 20'}),

        html.Div([
            dcc.Graph(
                id='3d-indicators-scatter-pred',
            )
        ], style={'display': 'inline-block', 'padding': '0 20'}),

        html.Div([
            html.P('Overall accuracy', style={'color': 'white'}),
            daq.LEDDisplay(
                id='accuracy-all',
                value='0000',
                color='red',
                size=50,
                style={'margin-bottom': '20px'}
            ),

            html.P('Accuracy on train data', style={'color': 'white'}),
            daq.LEDDisplay(
                id='accuracy-train',
                value='0000',
                color='red',
                size=50,
                style={'margin-bottom': '20px'}
            ),

            html.P('Accuracy on test data', style={'color': 'white'}),
            daq.LEDDisplay(
                id='accuracy-test',
                value='0000',
                color='red',
                size=50,
            )
        ], style={'display': 'inline-block', 'margin-left': '150px'}),
    ])
])


app.layout = html.Div(children=[
    html.Div(id='dark-theme-components', children=[
        daq.DarkThemeProvider(theme=theme, children=rootLayout)
    ], style={'backgroundColor': '#111111'})
])

@app.callback(
    Output('top5-countries', 'figure'),
    [Input('crossfilter-xaxis-column', 'value'),
     Input('date-picker-single', 'date'),
     Input('checklist', 'value'),
     Input('continent-1', 'value'),
     Input('continent-2', 'value'),
     Input('continent-3', 'value')]
)
def update_top5(indicator, date, checklist_val, cont1, cont2, cont3):
    if checklist_val == ['All']:
        dff = df.copy()
    else:
        dff = df[df['continent'].isin([cont1, cont2, cont3])]

    dff = dff[dff['date'] == date].sort_values(by=indicators_dict[indicator], ascending=False).head(5)
    fig = px.bar(dff, x=indicators_dict[indicator], y='iso_code', color='continent', hover_name='location', orientation='h',
                 template='plotly_dark', height=225, labels={'iso_code': '', indicators_dict[indicator]: ''},
                 title='Top 5 countries by '+indicator)
    fig.update_traces(hovertemplate=None)
    fig.update_layout(yaxis_categoryorder='total ascending', margin={'l': 20, 'b': 30, 'r': 10, 't': 30},
                      hovermode='x')

    return fig

@app.callback(
    Output('crossfilter-indicator-scatter', 'figure'),
    [Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value'),
     Input('date-picker-single', 'date'),
     Input('crossfilter-xaxis-type', 'value'),
     Input('crossfilter-yaxis-type', 'value'),
     Input('checklist', 'value'),
     Input('continent-1', 'value'),
     Input('continent-2', 'value'),
     Input('continent-3', 'value')]
)

def update_scatter(xaxis_column_name, yaxis_column_name, date_value, xaxis_type, yaxis_type, checklist_val, cont1, cont2, cont3):
    if checklist_val == ['All']:
        dff = df.copy()
    else:
        dff = df[df['continent'].isin([cont1, cont2, cont3])]

    dff = dff[dff['date'] == date_value]

    fig = px.scatter(dff, x=indicators_dict[xaxis_column_name], y=indicators_dict[yaxis_column_name], hover_name='location', size='population',
                     color='continent', template='plotly_dark')
    fig.update_traces(customdata=dff['location'])
    fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')
    fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')
    fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig

def create_time_series(dff, title, indicator):

    fig = px.scatter(dff, x='date', y=indicator, labels={'date': '', indicator: ''}, template='plotly_dark')
    fig.update_traces(mode='lines+markers')
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(type='linear')

    fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
                       xref='paper', yref='paper', showarrow=False, align='left',
                       bgcolor='rgba(255, 255, 255, 0.5)', text=title)
    fig.update_traces(hovertemplate=None)
    fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10}, hovermode='x')

    return fig

@app.callback(
    [Output('x-time-series', 'figure'),
     Output('y-time-series', 'figure')],
    [Input('countries', 'value'),
     Input('crossfilter-xaxis-column', 'value'),
     Input('crossfilter-yaxis-column', 'value')]
)

def update_graphs(country_name, indicator1, indicator2):
    dff = df[df['location'] == country_name]
    title = '<b>{}</b><br>{}'.format(country_name, indicator1)
    return create_time_series(dff, title, indicators_dict[indicator1]), create_time_series(dff, indicator2, indicators_dict[indicator2])


@app.callback(
    [Output('new-tests-led', 'value'),
     Output('new-cases-led', 'value'),
     Output('new-deaths-led', 'value'),
     Output('positive-rate', 'value')],
    [Input('countries', 'value'),
     Input('date-picker-single', 'date')]
)
def update_summary(country, date):
    tests = round(df[(df['location'] == country) & (df['date'] == date)]['new_tests_per_thousand'].item(), 2)
    cases = round(df[(df['location'] == country) & (df['date'] == date)]['new_cases_per_million'].item(), 2)
    deaths = round(df[(df['location'] == country) & (df['date'] == date)]['new_deaths_per_million'].item(), 2)
    positive_rate = round(df[(df['location'] == country) & (df['date'] == date)]['positive_rate'].item(), 2)
    if math.isnan(tests) or tests == 0:
        tests = '0000'
    if math.isnan(cases) or cases == 0:
        cases = '0000'
    if math.isnan(deaths) or deaths == 0:
        deaths = '0000'
    if math.isnan(positive_rate) or positive_rate == 0:
        positive_rate = '0000'
    return tests, cases, deaths, positive_rate

@app.callback(
    [Output('3d-indicators-scatter-actual', 'figure'),
     Output('3d-indicators-scatter-pred', 'figure'),
     Output('accuracy-all', 'value'),
     Output('accuracy-train', 'value'),
     Output('accuracy-test', 'value')],
    [Input('date-picker-single', 'date')]
)
def plot_model_clusters(date):
    cont_list = ['Europe', 'Asia', 'North America']
    return plot_clusters(df, cont_list, vars, date)

if __name__ == '__main__':
    app.run_server(debug=True)