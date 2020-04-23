import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction

import numpy as np
import pandas as pd
from scipy.stats import binom, poisson # binomial and poisson distribution functions

import datetime
from datetime import datetime as dt
import pathlib

import plotly.graph_objects as go

import datetime # library for date-time functionality
import model_fxns as fxns
from time import sleep

import urllib.parse
import flask
import io
import sys 

#from io import StringIO
#import base64 # functionality for encoding binary data to ASCII characters and decoding back to binary data
#from IPython.display import HTML, display

#global df_for_download

col_names =  ['obs_y', 'pred_y', 'forecasted_y', 'pred_dates', 'forecast_dates', 'label', 'obs_pred_r2', 'model', 
              'focal_loc', 'PopSize', 'ArrivalDate', 'pred_clr', 'fore_clr']

fits_df = pd.DataFrame(columns = col_names)


app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],)

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

server = app.server
app.config.suppress_callback_exceptions = True


# Path
BASE_PATH = pathlib.Path(__file__).parent.resolve()
DATA_PATH = BASE_PATH.joinpath("data").resolve()


# Read data
#try:
#    seir_fits_df = pd.read_csv('https://raw.githubusercontent.com/Rush-Quality-Analytics/SupplyDemand/master/notebooks/data/SEIR-SD_States.txt', sep='\t')
#except:
seir_fits_df = pd.read_csv('data/SEIR-SD_States.txt', sep='\t')


#try: 
#    path = 'https://raw.githubusercontent.com/Rush-Quality-Analytics/SupplyDemand/master/notebooks/data/StatePops.csv'
#    statepops = pd.read_csv('path')

#except:
statepops = pd.read_csv('data/StatePops.csv')

    
#try: 
#    ap_df = pd.read_csv('https://raw.githubusercontent.com/Rush-Quality-Analytics/SupplyDemand/master/notebooks/data/COVID-CASES-DF.txt', sep='\t') 
#    ap_df = ap_df[ap_df['Country/Region'] == 'US']
#    ap_df = ap_df[ap_df['Province/State'] != 'US']
#    ap_df = ap_df[ap_df['Province/State'] != 'American Samoa']
#    ap_df = ap_df[ap_df['Province/State'] != 'Northern Mariana Islands']
#    ap_df.drop(columns=['Unnamed: 0'], inplace=True)
#except:
locs_df = pd.read_csv('data/COVID-CASES-DF.txt', sep='\t') 
locs_df = locs_df[locs_df['Country/Region'] == 'US']
locs_df = locs_df[locs_df['Province/State'] != 'US']
locs_df = locs_df[locs_df['Province/State'] != 'American Samoa']
locs_df = locs_df[locs_df['Province/State'] != 'Northern Mariana Islands']
locs_df.drop(columns=['Unnamed: 0'], inplace=True)

locations = list(set(locs_df['Province/State']))


models = ['Logistic', 'Gaussian', 'SEIR-SD', '3rd degree polynomial', 'Quadratic', 'Exponential']
day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
            'Friday', 'Saturday','Sunday']



def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Center for Quality, Safety and Value Analytics", style={
            'textAlign': 'left',
            'color': '#2F9314'
        }),
            html.Div(
                id="intro",
                children="Obtain forecasts for COVID-19 using a suite of models.",
            ),
        ],
    )



def generate_control_card():
    
    """
    :return: A Div containing controls for graphs.
    """
    return html.Div(
        id="control-card",
        children=[
            html.P("Select Location"),
            dcc.Dropdown(
                id="location-select",
                options=[{"label": i, "value": i} for i in locations],
                value='Illinois',
                style={
                    #'height': '2px', 
                    'width': '250px', 
                    'font-size': "100%",
                    #'min-height': '1px',
                    }
            ),
            html.Br(),
            html.P("Select Model"),
            dcc.Dropdown(
                id="model-select",
                options=[{"label": i, "value": i} for i in models],
                value='SEIR-SD',
            ),
            html.Br(),
            html.P("% of cases visiting your hospital"),
            dcc.Slider(
                id="visits",
                min=0,
                max=100,
                step=1,
                value=10,
                marks={
                    0: '0',
                    10: '10',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50',
                    60: '60',
                    70: '70',
                    80: '80',
                    90: '90',
                    100: '100'
                    },
            ),    
            html.Br(),
            html.P("% of visits resulting in admission"),
            dcc.Slider(
                id="admits",
                min=0,
                max=100,
                value=20,
                step=1,
                marks={
                    0: '0',
                    10: '10',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50',
                    60: '60',
                    70: '70',
                    80: '80',
                    90: '90',
                    100: '100'
                    },
                
            ),   
            html.Br(),
            html.P("% of admits going to ICU"),
            dcc.Slider(
                id="percent ICU",
                min=0,
                max=100,
                value=20,
                step=1,
                marks={
                    0: '0',
                    10: '10',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50',
                    60: '60',
                    70: '70',
                    80: '80',
                    90: '90',
                    100: '100'
                    },
                
            ),   
            html.Br(),
            html.P("% of ICU on ventilator"),
            dcc.Slider(
                id="on vent",
                min=0,
                max=100,
                value=20,
                step=1,
                marks={
                    0: '0',
                    10: '10',
                    20: '20',
                    30: '30',
                    40: '40',
                    50: '50',
                    60: '60',
                    70: '70',
                    80: '80',
                    90: '90',
                    100: '100'
                    },
                
            ),   
            html.Br(),
            html.P("Avg LOS for non-ICU"),
            dcc.Slider(
                id="non-ICU LOS",
                min=1,
                max=14,
                value=4,
                step=1,
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9',
                    10: '10',
                    11: '11',
                    12: '12',
                    13: '13 days'
                    
                    },
                
            ),   
            html.Br(),
            html.P("Avg LOS for ICU"),
            dcc.Slider(
                id="ICU LOS",
                min=1,
                max=14,
                value=4,
                step=1,
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9',
                    10: '10',
                    11: '11',
                    12: '12',
                    13: '13 days'
                    
                    },
                
            ),
            html.Br(),
            html.P("Avg time lag in patient visits"),
            dcc.Slider(
                id="time lag",
                min=1,
                max=14,
                value=4,
                step=1,
                marks={
                    1: '1',
                    2: '2',
                    3: '3',
                    4: '4',
                    5: '5',
                    6: '6',
                    7: '7',
                    8: '8',
                    9: '9',
                    10: '10',
                    11: '11',
                    12: '12',
                    13: '13 days'
                    
                    },
                
            ),   
            html.Br(),
            html.Div(
                id="reset-btn-outer",
                children=html.Button(id="reset-btn", children="Reset", n_clicks=0),
            ),
        ],
    )





def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )



def generate_model_forecast_plot(loc,  model, reset):
    
    
    new_cases = []
    ForecastDays = 60
    
    col_names =  ['obs_y', 'pred_y', 'forecasted_y', 'pred_dates', 'forecast_dates', 'label', 'obs_pred_r2', 'model', 
                      'focal_loc', 'PopSize', 'ArrivalDate', 'pred_clr', 'fore_clr']
        
    fits_df  = pd.DataFrame(columns = col_names)

    PopSize = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
    PopSize = PopSize[0]
        
    ArrivalDate = statepops[statepops['Province/State'] == loc]['Date_of_first_reported_infection'].tolist()
    ArrivalDate = ArrivalDate[0]
        
    SEIR_Fit = seir_fits_df[seir_fits_df['focal_loc'] == loc]
        
        
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
        
        
    # filter main dataframe to include only the chosen location
    df_sub = locs_df[locs_df['Province/State'] == loc]
        
    # get column labels, will filter below to extract dates
    yi = list(df_sub)
        
        
        
    obs_y_trunc = []
    fore_clrs =  ['purple',  'mediumorchid', 'plum', 'blue', 'deepskyblue', 'darkturquoise',
                  'green', 'limegreen', 'gold', 'orange', 'red']
    pred_clrs = ['0.0', '0.1', '0.2', '0.25', '0.3', '0.35', '0.4', '0.5',
                 '0.6', '0.7', '0.8']
        
        
    for i, j in enumerate(list(range(-10, 1))):
        pred_clr = pred_clrs[i]
        fore_clr = fore_clrs[i]
            
        if j == 0:
            # get dates for today's predictions/forecast
            DATES = yi[4:]
            obs_y_trunc = df_sub.iloc[0,4:].values
        else:
            # get dates for previous days predictions/forecast
            DATES = yi[4:j]
            obs_y_trunc = df_sub.iloc[0,4:j].values
            
            
        ii = 0
        while obs_y_trunc[ii] == 0: ii+=1
        y = obs_y_trunc[ii:]
        dates = DATES[ii:]
            
    
        # declare x as a list of integers from 0 to len(y)
        x = list(range(len(y)))

        # Call function to use chosen model to obtain:
        #    r-square for observed vs. predicted
        #    predicted y-values
        #    forecasted x and y values
        iterations = 2
        obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = fxns.fit_curve(x, y, 
                            model, ForecastDays, PopSize, ArrivalDate, j, iterations, SEIR_Fit)
            
        # convert y values to numpy array
        y = np.array(y)

        # because it isn't based on a best fit line,
        # and the y-intercept is forced through [0,0]
        # a model can perform so poorly that the 
        # observed vs predicted r-square is negative (a nonsensical value)
        # if this happens, report the r-square as 0.0
        if obs_pred_r2 < 0:
            obs_pred_r2 = 0.0

        # convert any y-values (observed, predicted, or forecasted)
        # that are less than 0 (nonsensical values) to 0.
        y[y < 0] = 0
        pred_y = np.array(pred_y)
        pred_y[pred_y < 0] = 0

        forecasted_y = np.array(forecasted_y)
        forecasted_y[forecasted_y < 0] = 0
            
        # number of from ArrivalDate to end of forecast window
        #numdays = len(forecasted_x)
        latest_date = pd.to_datetime(dates[-1])
        first_date = pd.to_datetime(dates[0])

        # get the date of the last day in the forecast window
        future_date = latest_date + datetime.timedelta(days = ForecastDays-1)
            
        # get all dates from ArrivalDate to the last day in the forecast window
        fdates = pd.date_range(start=first_date, end=future_date)
        fdates = fdates.strftime('%m/%d')
            
        # designature plot label for legend
        if j == 0:
            label='Current forecast'
            
        else:
            label = str(-j)+' day old forecast'
            
            
        if label == 'Current forecast':
            for i, val in enumerate(forecasted_y):
                if i > 0:
                    if forecasted_y[i] - forecasted_y[i-1] > 0:
                        new_cases.append(forecasted_y[i] - forecasted_y[i-1])
                    else:
                        new_cases.append(0)
                if i == 0:
                    new_cases.append(forecasted_y[i])
                        
                
        # get dates from ArrivalDate to the current day
        dates = pd.date_range(start=first_date, end=latest_date)
        dates = dates.strftime('%m/%d')
            
            
        output_list = [y, pred_y, forecasted_y, dates, fdates,
                       label, obs_pred_r2, model, loc, PopSize, 
                       ArrivalDate, pred_clr, fore_clr]
            
        fits_df.loc[len(fits_df)] = output_list
        
    
    labels = fits_df['label'].tolist()
        
    fig_data = []
    
    for i, label in enumerate(labels):
            
        sub_df = fits_df[fits_df['label'] == label]
        
        dates = sub_df['pred_dates'].iloc[0]
        clr = sub_df['pred_clr'].iloc[0]
        obs_y = sub_df['obs_y'].iloc[0]
        if i == 0:
            fig_data.append(
                go.Scatter(
                    x=dates,
                    y=obs_y,
                    mode="markers",
                    name='Observed',
                    opacity=0.75,
                    marker=dict(color='#243220', size=10)
                )
            )
        
        
        fdates = sub_df['forecast_dates'].iloc[0]
        forecasted_y = sub_df['forecasted_y'].iloc[0]
        clr = sub_df['fore_clr'].iloc[0]
        #focal_loc = sub_df['focal_loc'].iloc[0]
        #popsize = sub_df['PopSize'].iloc[0]
            
        pred_y = sub_df['pred_y'].iloc[0]
        # plot forecasted y values vs. dates
        l = int(len(pred_y)+ForecastDays)
            
        forecasted_y = forecasted_y[0 : l]
        fdates = fdates[0 : l]
        
        fig_data.append(
            go.Scatter(
                x=fdates,
                y=forecasted_y,
                name=label,
                mode="lines",
                line=dict(color=clr, width=2)
            )
        )
        
  
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>Number of COVID-19 cases</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=300,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    #sleep(0.0)
    
    return figure

            
        
        
        
        
def generate_patient_census_plot(loc,  model, per_loc, per_admit, 
    per_cc, LOS_cc, LOS_nc, per_vent, TimeLag, reset):
    
    global df_for_download
    
    new_cases = []
    ForecastDays = 60
    
    PopSize = statepops[statepops['Province/State'] == loc]['PopSize'].tolist()
    PopSize = PopSize[0]
        
    ArrivalDate = statepops[statepops['Province/State'] == loc]['Date_of_first_reported_infection'].tolist()
    ArrivalDate = ArrivalDate[0]
        
    SEIR_Fit = seir_fits_df[seir_fits_df['focal_loc'] == loc]
      
    # add 1 to number of forecast days for indexing purposes
    ForecastDays = int(ForecastDays+1)
        
        
    # filter main dataframe to include only the chosen location
    df_sub = locs_df[locs_df['Province/State'] == loc]
        
    # get column labels, will filter below to extract dates
    yi = list(df_sub)
        
    obs_y_trunc = []
    DATES = yi[4:]
    obs_y_trunc = df_sub.iloc[0,4:].values
    

    ii = 0
    while obs_y_trunc[ii] == 0: ii+=1
    y = obs_y_trunc[ii:]
    dates = DATES[ii:]
            
    
    # declare x as a list of integers from 0 to len(y)
    x = list(range(len(y)))

    # Call function to use chosen model to obtain:
    #    r-square for observed vs. predicted
    #    predicted y-values
    #    forecasted x and y values
    iterations = 2
    obs_pred_r2, obs_x, pred_y, forecasted_x, forecasted_y, params = fxns.fit_curve(x, y, 
                            model, ForecastDays, PopSize, ArrivalDate, 0, iterations, SEIR_Fit)
            
    # convert y values to numpy array
    y = np.array(y)

    # because it isn't based on a best fit line,
    # and the y-intercept is forced through [0,0]
    # a model can perform so poorly that the 
    # observed vs predicted r-square is negative (a nonsensical value)
    # if this happens, report the r-square as 0.0
    if obs_pred_r2 < 0:
        obs_pred_r2 = 0.0

    # convert any y-values (observed, predicted, or forecasted)
    # that are less than 0 (nonsensical values) to 0.
    y[y < 0] = 0
    forecasted_y = np.array(forecasted_y)
    forecasted_y[forecasted_y < 0] = 0
            
    # number of from ArrivalDate to end of forecast window
    #numdays = len(forecasted_x)
    latest_date = pd.to_datetime(dates[-1])
    first_date = pd.to_datetime(dates[0])

    # get the date of the last day in the forecast window
    future_date = latest_date + datetime.timedelta(days = ForecastDays-1)
    
    # get all dates from ArrivalDate to the last day in the forecast window
    fdates = pd.date_range(start=first_date, end=future_date)
    fdates = fdates.strftime('%m/%d')
            
    # designature plot label for legend
    for i, val in enumerate(forecasted_y):
        if i > 0:
            if forecasted_y[i] - forecasted_y[i-1] > 0:
                new_cases.append(forecasted_y[i] - forecasted_y[i-1])
            else:
                new_cases.append(0)
        if i == 0:
            new_cases.append(forecasted_y[i])
                        
                
    # get dates from ArrivalDate to the current day
    dates = pd.date_range(start=first_date, end=latest_date)
    dates = dates.strftime('%m/%d')
            
    # declare column labels
    col_labels = ['Total cases', 'New cases', 'New visits', 'New admits',
                  'All COVID', 'Non-ICU', 'ICU', 'Vent']
    
    # row labels are the dates
    row_labels = fdates.tolist()
        
    #### Inclusion of time lag
    # time lag is modeled as a Poisson distributed 
    # random variable with a mean chosen by the user (TimeLag)
    new_cases_lag = []
    x = list(range(len(forecasted_y)))
    for i in new_cases:
        lag_pop = i*poisson.pmf(x, TimeLag)
        new_cases_lag.append(lag_pop)
         
    # Declare a list to hold time-staggered lists
    # This will allow the time-lag effects to
    # be summed across rows (days)
    lol = []
    for i, daily_vals in enumerate(new_cases_lag):
        # number of indices to pad in front
        fi = [0]*i
        diff = len(new_cases) - len(fi)
        # number of indices to pad in back
        bi = [0]*diff
        ls = list(fi) + list(daily_vals) + list(bi)
        lol.append(np.array(ls))
        
    # convert the list of time-staggered lists to an array
    ar = np.array(lol)
        
    # get the time-lagged sum of visits across days
    ts_lag = np.sum(ar, axis=0)
    # upper truncate for the number of days in observed y values
    ts_lag = ts_lag[:len(new_cases)]
        
    # row labels are the dates
    row_labels = fdates.tolist()  
        
    # Declare pandas dataframe to hold data for download
    df_for_download = pd.DataFrame(columns = ['date'] + col_labels)
    
    #### Construct arrays for critical care and non-critical care patients
    cc = (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc) * np.array(ts_lag)
    cc = cc.tolist()
        
    nc = (1 - (0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc) * np.array(ts_lag)
    nc = nc.tolist()
        
    # Model length of stay (LOS) as a binomially distributed
    # random variable according to binomial parameters p and n
    #    p: used to obtain a symmetrical distribution 
    #    n: (n_cc & n_nc) = 2 * LOS will produce a binomial
    #       distribution with a mean equal to the LOS
        
    p = 0.5
    n_cc = LOS_cc*2
    n_nc = LOS_nc*2
        
    # get the binomial random variable properties
    rv_nc = binom(n_nc, p)
    # Use the binomial cumulative distribution function
    p_nc = rv_nc.cdf(np.array(range(1, len(fdates)+1)))
        
    # get the binomial random variable properties
    rv_cc = binom(n_cc, p)
    # Use the binomial cumulative distribution function
    p_cc = rv_cc.cdf(np.array(range(1, len(fdates)+1)))
        
    # Initiate lists to hold numbers of critical care and non-critical care patients
    # who are expected as new admits (index 0), as 1 day patients, 2 day patients, etc.
    LOScc = np.zeros(len(fdates))
    LOScc[0] = ts_lag[0] * (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc)
    LOSnc = np.zeros(len(fdates))
    LOSnc[0] =  ts_lag[0] * (1-(0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc)
        
    total_nc = []
    total_cc = []
    
    #print(len(fdates), len(row_labels), len(ts_lag), len(new_cases)) 
    #import sys
    #sys.exit()
        
    # Roll up patient carry-over into lists of total critical care and total
    # non-critical patients expected
    for i, day in enumerate(fdates):
        LOScc = LOScc * (1 - p_cc)
        LOSnc = LOSnc * (1 - p_nc)
            
        LOScc = np.roll(LOScc, shift=1)
        LOSnc = np.roll(LOSnc, shift=1)
            
        LOScc[0] = ts_lag[i] * (0.01 * per_cc) * (0.01 * per_admit) * (0.01 * per_loc)
        LOSnc[0] = ts_lag[i] * (1 - (0.01 * per_cc)) * (0.01 * per_admit) * (0.01 * per_loc)
    
        total_nc.append(np.sum(LOSnc))
        total_cc.append(np.sum(LOScc))
        
    
    
    for i in range(len(row_labels)):
            
        new = new_cases[i]
        val = ts_lag[i]
            
        # each cell is a row with 4 columns:
        #     Total cases, 
        #     new cases, 
        #     time-lagged visits to your hospital,
        #     time-lagged admits to your hospital
        cell = [int(np.round(forecasted_y[i])), 
                int(np.round(new)), 
                int(np.round(val * (per_loc * 0.01))),
                int(np.round((0.01 * per_admit) * val * (per_loc * 0.01))),
                int(np.round(total_nc[i] + total_cc[i])), 
                int(np.round(total_nc[i])),
                int(np.round(total_cc[i])), 
                int(np.round(total_cc[i]*(0.01*per_vent)))]
        
            
        # Add the row to the dataframe
        df_row = [row_labels[i]]
        df_row.extend(cell)
        labs = ['date'] + col_labels
        temp = pd.DataFrame([df_row], columns=labs)
        df_for_download = pd.concat([df_for_download, temp])
        # color the first row grey and remaining rows white
        
    
    labels = list(df_for_download)
    labels = labels[1:]
    fig_data = []
    
    #print(len(labels))
    #sys.exit()
    
    clrs = ['purple',  'mediumorchid', 'blue', 'deepskyblue',
            'green', 'gold', 'orange', 'red']
    
    for i, label in enumerate(labels):
        if label == 'date' or label == 'Total cases' or label == 'New cases':
            continue
        
        dates = df_for_download['date'].tolist()
        clr = clrs[i]
        obs_y = df_for_download[label].tolist()
        
        fig_data.append(
            go.Scatter(
                x=dates,
                y=obs_y,
                mode="lines",
                name=label,
                opacity=0.75,
                line=dict(color=clr, width=2)
            )
        )
        
    
    figure = go.Figure(
        data=fig_data,
        layout=go.Layout(
            xaxis=dict(
                title=dict(
                    text="<b>Date</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            yaxis=dict(
                title=dict(
                    text="<b>Number of COVID-19 cases</b>",
                    font=dict(
                        family='"Open Sans", "HelveticaNeue", "Helvetica Neue",'
                        " Helvetica, Arial, sans-serif",
                        size=18,
                        
                    ),
                ),
                rangemode="tozero",
                zeroline=True,
                showticklabels=True,
            ),
            
            margin=dict(l=60, r=30, b=10, t=40),
            showlegend=True,
            height=300,
            paper_bgcolor="rgb(245, 247, 249)",
            plot_bgcolor="rgb(245, 247, 249)",
        ),
    )
    #sleep(0.0)
    
    return figure



        
        









app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("RUSH_full_color.jpg"), style={'textAlign': 'left'}),
                      html.Img(src=app.get_asset_url("plotly_logo.png"), style={'textAlign': 'right'})],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="three columns",
            children=[description_card(), generate_control_card()]
            + [
                html.Div(
                    ["initial child"], id="output-clientside", style={"display": "none"}
                )
            ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="nine columns",
            children=[
                
                # Plot of model forecast
                html.Div(
                    id="model_forecasts",
                    style={'fontSize':26},
                    children=[
                        html.B("Model Forecasts"),
                        html.Hr(),
                        dcc.Graph(id="model_forecasts_plot"),
                    ],
                ),
                html.Div(
                    id="patient_census",
                    style={'fontSize':26},
                    children=[
                        html.B("Forecasted Patient Census"),
                        html.Hr(),
                        dcc.Graph(id="patient_census_plot"),
                    ],
                ),
                
            ],
        ),
        html.Div(
            id='field-dropdown',
            className='two columns',
            children=[
            html.Div(id='table'),
            html.A(html.Button('Export to Excel'),
                        id='download_xlsx'),]


)
    ],
)




@app.callback(
    Output("model_forecasts_plot", "figure"),
    [
        Input("location-select", "value"),
        Input("model-select", "value"),
        Input("reset-btn", "n_clicks"),
    ],
)

def update_model_forecast(loc, model, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return generate_model_forecast_plot(loc, model, reset)



@app.callback(
    Output("patient_census_plot", "figure"),
    
    [Input("location-select", "value"),
     Input("model-select", "value"),
     Input("visits",  "value"),
     Input("admits",  "value"),
     Input("percent ICU", "value"),
     Input("ICU LOS", "value"),
     Input("non-ICU LOS", "value"),
     Input("on vent",  "value"),
     Input("time lag", "value"),
     Input("reset-btn", "n_clicks"),
    ],
)

def update_patient_census(loc,  model, per_loc, per_admit, 
    per_cc, LOS_cc, LOS_nc, per_vent, TimeLag, reset_click):
    
    reset = False
    # Find which one has been triggered
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "reset-btn":
            reset = True

    # Return to original hm(no colored annotation) by resetting
    return generate_patient_census_plot(loc,  model, per_loc, per_admit, 
    per_cc, LOS_cc, LOS_nc, per_vent, TimeLag, reset)




@app.callback(
    Output('table', 'children'))
def update_table():
    
    return generate_table(df_for_download)


@app.callback(
    Output('download_xlsx', 'href'),
    [Input('field-dropdown', 'value')])
def update_download_link(filter_value):
    return f'/export/excel'

@app.server.route('/export/excel')
def export_excel_file():
    option_df = df_for_download
    xlsx_io = io.BytesIO()
    writer = pd.ExcelWriter(xlsx_io, engine='xlsxwriter')
    option_df.to_excel(writer, sheet_name='scheme', index=False)
    writer.save()
    xlsx_io.seek(0)

    return flask.send_file(
        xlsx_io,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        attachment_filename=f'export.xlsx',
        as_attachment=True,
        cache_timeout=0
    )




# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
