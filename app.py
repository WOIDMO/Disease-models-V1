import dash_table
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc


############################################ the model ################################################
def deriv(y, t, N, p_immune, beta, gamma, delta, p_I_to_C, T_I_to_C, p_C_to_D, T_C_to_R, T_C_to_D):
    S, E, I, C, R, D = y
    dSdt = -beta * I * S * (1 - p_immune) / N
    dEdt = beta * I * S * (1 - p_immune) / N - delta * E
    dIdt = delta * E - 1 / T_I_to_C * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    dCdt = 1 / T_I_to_C * p_I_to_C * I - 1 / T_C_to_D * p_C_to_D * C - (1 - p_C_to_D) * 1 / T_C_to_R * C
    dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1 / T_C_to_R * C
    dDdt = 1 / T_C_to_D * p_C_to_D * C
    return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt


def Model(initial_cases, initial_date, N, p_immune, beta, gamma, delta, p_I_to_C, T_I_to_C, p_C_to_D, T_C_to_R,
          T_C_to_D):
    days = 365
    y0 = N - initial_cases, initial_cases, 0.0, 0.0, 0.0, 0.0
    t = np.linspace(1, days, days)  # ORN changed start = 0 to start = 1 for integer values
    #    print(t)
    ret = odeint(deriv, y0, t, args=(N, p_immune, beta, gamma, delta, p_I_to_C, T_I_to_C, p_C_to_D, T_C_to_R, T_C_to_D))
    # todo: make sure to find a way to implement the critical community size truncating - Actually, Steve said no need
    S, E, I, C, R, D = ret.T
    # note this is actually IFR: Dead / total_infected
    total_CFR = [0] + [100 * D[i] / sum(delta * E[:i]) if sum(
        delta * E[:i]) > 0 else 0 for i in range(1, len(t))]
    daily_CFR = [0] + [100 * ((D[i] - D[i - 1]) / ((R[i] - R[i - 1]) + (D[i] - D[i - 1]))) if max(
        (R[i] - R[i - 1]), (D[i] - D[i - 1])) > 10 else 0 for i in range(1, len(t))]
    return t, S, E, I, C, R, D, total_CFR, daily_CFR


############### initialization ###############


population = 100000  # size of initial population
initial_cases = 10  # initial number of Exposed. I chose 10 to stay away from the critical community size at the onset of the epidemic.

num_reps = 3  # for Monte Carlo; num_reps = 1000 is a good number but takes a long time.
# num_reps = 200 is good for the Median graphs,
# and also gives a good sense on how the histogram will look like.

disease = 'deterministic_test'  # supported options: 'Covid_19', 'Influenza_seasonal', or 'deterministic_test'
print(disease)

# The section below is intended to be modular, so we can easily add other disease model.
if disease == 'Covid_19':
    R_0_avg, R_0_err = 3.82, 2.08
    p_immune_avg, p_immune_err = 6 / 100, 1 / 100
    T_infectious_avg, T_infectious_err = 10, 2
    T_incubation_avg, T_incubation_err = 5.5, 2.5
    p_I_to_C_avg, p_I_to_C_err = 5 / 100, 3 / 100
    p_C_to_D_avg, p_C_to_D_err = 32.5 / 100, 7.5 / 100
    T_I_to_C_avg, T_I_to_C_err = 8.5, 4
    T_C_to_D_avg, T_C_to_D_err = 7, 2
    T_C_to_R_avg, T_C_to_R_err = 14, 2
elif disease == 'Influenza_seasonal':
    R_0_avg, R_0_err = 1.5, 0.6
    p_immune_avg, p_immune_err = 20 / 100, 5 / 100
    T_infectious_avg, T_infectious_err = 6.5, 3.5
    T_incubation_avg, T_incubation_err = 2.5, 1.5
    p_I_to_C_avg, p_I_to_C_err = 0.5 / 100, 0.25 / 100
    p_C_to_D_avg, p_C_to_D_err = 40.5 / 100, 7.5 / 100
    T_I_to_C_avg, T_I_to_C_err = 8.5, 4
    T_C_to_D_avg, T_C_to_D_err = 7, 2
    T_C_to_R_avg, T_C_to_R_err = 14, 2
elif disease == 'deterministic_test':
    R_0_avg, R_0_err = 3.0, 0
    p_immune_avg, p_immune_err = 0, 0
    T_infectious_avg, T_infectious_err = 9, 0
    T_incubation_avg, T_incubation_err = 3, 0
    p_I_to_C_avg, p_I_to_C_err = 5 / 100, 0
    p_C_to_D_avg, p_C_to_D_err = 30 / 100, 0
    T_I_to_C_avg, T_I_to_C_err = 12, 0
    T_C_to_D_avg, T_C_to_D_err = 7.5, 0
    T_C_to_R_avg, T_C_to_R_err = 6.5, 0
else:
    print('undefined disease!')

# Now roll the dice num_reps times for each parameter:
p_immune = np.random.uniform(p_immune_avg - p_immune_err, p_immune_avg + p_immune_err, num_reps).round(
    3)  # portion of immuned population
R_0 = np.random.uniform(R_0_avg - R_0_err, R_0_avg + R_0_err, num_reps).round(2)  # Initial R0
T_infectious = np.random.uniform(T_infectious_avg - T_infectious_err, T_infectious_avg + T_infectious_err,
                                 num_reps).round(2)  # infectious period [days]
T_incubation = np.random.uniform(T_incubation_avg - T_incubation_err, T_incubation_avg + T_incubation_err,
                                 num_reps).round(2)  # incubation period [days] - not contagious during this time
T_I_to_C = np.random.uniform(T_I_to_C_avg - T_I_to_C_err, T_I_to_C_avg + T_I_to_C_err, num_reps).round(
    2)  # time to transition from Infected to Critical [days]
T_C_to_D = np.random.uniform(T_C_to_D_avg - T_C_to_D_err, T_C_to_D_avg + T_C_to_D_err, num_reps).round(
    2)  # time to transition from Critical to Dead [days]
T_C_to_R = np.random.uniform(T_C_to_R_avg - T_C_to_R_err, T_C_to_R_avg + T_C_to_R_err, num_reps).round(
    2)  # time to transition from Critical to Recovered [days]
p_I_to_C = np.random.uniform(p_I_to_C_avg - p_I_to_C_err, p_I_to_C_avg + p_I_to_C_err, num_reps).round(
    3)  # overall probability of transition from Infected to Critical
p_C_to_D = np.random.uniform(p_C_to_D_avg - p_C_to_D_err, p_C_to_D_avg + p_C_to_D_err, num_reps).round(
    3)  # overall probability of transition from Infected to Critical

gamma = 1.0 / T_infectious
delta = 1.0 / T_incubation
beta = R_0 / T_infectious
initial_date = 0

############### Monte Carlo ###############
print('start Monte-Carlo')
df_results = pd.DataFrame({})  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
df_I_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
I_vs_t = []
df_C_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
C_vs_t = []
df_D_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
D_vs_t = []
df_IFR_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
IFR_vs_t = []
for i in range(num_reps):
    time, S, E, I, C, R, D, total_CFR, daily_CFR = Model(initial_cases, initial_date, population, p_immune[i], beta[i],
                                                         gamma[i],
                                                         delta[i], p_I_to_C[i], T_I_to_C[i], p_C_to_D[i], T_C_to_R[i],
                                                         T_C_to_D[i])
    df_temp = pd.DataFrame(index=[i], data={'R_0': R_0[i], 'immune %': p_immune[i] * 100,
                                            'T_infectious': T_infectious[i], 'T_incubation': T_incubation[i],
                                            'p_I_to_C%': p_I_to_C[i] * 100, 'p_C_to_D%': p_C_to_D[i] * 100,
                                            'T_I_to_C': T_I_to_C[i], 'T_C_to_R': T_C_to_R[i], 'T_C_to_D': T_C_to_D[i],
                                            'peak E': max(E), 'peak I': max(I), 'peak R': max(R),
                                            'peak C': max(C), 'max D': max(D), 'total IFR': max(total_CFR)})
    df_results = df_results.append(df_temp)
    # plot progress to terminal, to avoid the feeling it got stuck...
    if round(i / 25, 0) == (i / 25):
        print('in MC round ', i, 'out of ', num_reps)

    # thanks to
    # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop for this method
    IFR_vs_t.append(total_CFR)
    I_vs_t.append(I)
    D_vs_t.append(D)
    C_vs_t.append(C)
    df_IFR_vs_t = pd.DataFrame(IFR_vs_t)  # compartment value for each day is appended as new row
    df_I_vs_t = pd.DataFrame(I_vs_t)  # compartment value for each day is appended as new row
    df_C_vs_t = pd.DataFrame(C_vs_t)  # compartment value for each day is appended as new row
    df_D_vs_t = pd.DataFrame(D_vs_t)  # compartment value for each day is appended as new row

########## print some results to the Terminal ###################
# Set it to None to display all columns in the dataframe,
# but actually print to terminal only if <15 rows (15 being arbitrary, used for testing)
if num_reps < 15:
    pd.set_option('display.max_columns', None)
    print(df_results)
else:
    print('NOT printing df_results to Terminal, as it is deemed too long')
print('min / max I')
print(min(df_results['peak I']), max(df_results['peak I']))
print('min / max C')
print(min(df_results['peak C']), max(df_results['peak C']))
print('min / max D')
print(min(df_results['max D']), max(df_results['max D']))
print('min(IFR), max(IFR)')
print('min / max R')
print(min(df_results['peak R']), max(df_results['peak R']))
print(min(df_results['total IFR']), max(df_results['total IFR']))

#############  try using dash  ###############


#  plot using ploty express, with the intention to transfer it to DCC as explained in https://dash.plotly.com/dash-core-components/graph
#  could not make it run with the regular dcc.Graph( ...  id='example-graph', ... figure={ ...  'data': [
#  in that case it worked for a single line but I couldn't figure out how to pull all lines from the DataFrame. It kept failing to load the web page.

# First handful or so runs of Monte Carlo graphs: Compartment content vs. time
fig1 = px.line(df_I_vs_t[1:20].T,
               title="Number of Infected vs. time, per Monte Carlo run (1st 20)",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Number of Infected"
               },
               )
fig1.layout.update(showlegend=False)  # eliminate legends as they can be very long (one for each Monte Carlo run)

fig2 = px.line(df_C_vs_t[1:20].T,
               title="Number of Critical vs. time, per Monte Carlo run (1st 20)",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Number of Critical"
               },
               )
fig2.layout.update(showlegend=False)  # eliminate legends as they can be very long (one for each Monte Carlo run)

fig3 = px.line(df_D_vs_t[1:20].T,
               title="Number of Dead vs. time, per Monte Carlo run (1st 20)",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Number of Dead"
               },
               )
fig3.layout.update(showlegend=False)  # eliminate legends as they can be very long (one for each Monte Carlo run)
# Histograms
fig4 = px.histogram(df_results, x="total IFR",
                    title="Histogram of IFR (Infected / Dead) over all Monte Carlo runs",
                    width=600, height=400,
                    )
fig5 = px.histogram(df_results, x="peak C",
                    title="Histogram of maximum number of Critical over all Monte Carlo runs",
                    width=600, height=400,
                    )
fig6 = px.histogram(df_results, x="max D",
                    title="Histogram of final Dead number over all Monte Carlo runs",
                    width=600, height=400,
                    )
# Median graphs
oneSigmaI = df_I_vs_t.T.apply(np.std, axis=1)
medianI = df_I_vs_t.T.apply(np.median, axis=1)
medianI_df = pd.DataFrame(data={'median': medianI, 'median+STD': medianI + oneSigmaI})
fig7 = px.line(medianI_df,
               title="Median of Infected vs. time, over all Monte Carlo run",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Median of Infected"
               },
               )

oneSigmaC = df_C_vs_t.T.apply(np.std, axis=1)
medianC = df_C_vs_t.T.apply(np.median, axis=1)
medianRangeC_df = pd.DataFrame(data={'median': medianC, 'median+STD': medianC + oneSigmaC})
fig8 = px.line(medianRangeC_df,
               title="Median of Critical vs. time, over all Monte Carlo run",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Median of Critical"
               },
               )

oneSigmaD = df_D_vs_t.T.apply(np.std, axis=1)
medianD = df_D_vs_t.T.apply(np.median, axis=1)
medianRangeD_df = pd.DataFrame(data={'median': medianD, 'median+STD': medianD + oneSigmaD})
fig9 = px.line(medianRangeD_df,
               title="Median of Dead vs. time, over all Monte Carlo run",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Median of Dead"
               },
               )

oneSigmaIFR = df_IFR_vs_t.T.apply(np.std, axis=1)
medianIFR = df_IFR_vs_t.T.apply(np.median, axis=1)
medianRangeIFR_df = pd.DataFrame(
    data={'median': medianIFR, 'median+STD': medianIFR + oneSigmaIFR, 'median-STD': medianIFR - oneSigmaIFR})
fig10 = px.line(medianRangeIFR_df,
                title="Median of IFR vs. time, over all Monte Carlo run",
                width=600, height=400,
                labels={  # replaces default labels by column name
                    # ... thanks to https://plotly.com/python/styling-plotly-express/
                    "index": "time [days]", "value": "Median of IFR"
                },
                )

############################################ the dash app layout ################################################
external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "MonteCarlo"

# these are the controls where the parameters can be tuned.
# They are not placed on the screen here, we just define them.
# Each separate input (e.g. a slider for the fatality rate) is placed
# in its own "dbc.FormGroup" and gets a "dbc.Label" where we put its name.
# The sliders use the predefined "dcc.Slider"-class, the numeric inputs
# use "dbc.Input", etc., so we don't have to tweak anything ourselves.
# The controls are wrappen in a "dbc.Card" so they look nice.
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Number of Monte Carlo loops to run"),
                dbc.Input(
                    id="num_reps", type="number", placeholder="num_reps",
                    min=1, max=5000, step=1, value=num_reps,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Initial Cases"),
                dbc.Input(
                    id="initial_cases", type="number", placeholder="initial_cases",
                    min=1, max=1_000_000, step=1, value=initial_cases,
                )
            ]
        ),

        dbc.FormGroup(
            [
                dbc.Label("population"),
                dbc.Input(
                    id="population", type="number", placeholder="population",
                    min=10_000, max=1_000_000_000, step=10_000, value=population,
                )
            ]
        ),

        dbc.FormGroup(
            [
                dbc.Label("average fraction of population immuned"),
                dbc.Input(
                    id="p_immune_avg", type="number", placeholder="p_immune_avg",
                    min=0, max=1, step=0.005, value=p_immune_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- fraction of population immuned"),
                dbc.Input(
                    id="p_immune_err", type="number", placeholder="p_immune_err",
                    min=0, max=1, step=0.005, value=p_immune_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average R0"),
                dbc.Input(
                    id="R_0_avg", type="number", placeholder="R_0_avg",
                    min=1, max=10, step=0.05, value=R_0_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- R0"),
                dbc.Input(
                    id="R_0_err", type="number", placeholder="R_0_err",
                    min=0, max=10, step=0.05, value=R_0_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average probability of going to ICU when infected [fraction]"),
                dbc.Input(
                    id="p_I_to_C_avg", type="number", placeholder="p_I_to_C_avg",
                    min=0, max=1, step=0.001, value=p_I_to_C_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- probability of going to ICU when infected [fraction]"),
                dbc.Input(
                    id="p_I_to_C_err", type="number", placeholder="p_I_to_C_err",
                    min=0, max=1, step=0.001, value=p_I_to_C_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average probability of dying in ICU [fraction]"),
                dbc.Input(
                    id="p_C_to_D_avg", type="number", placeholder="p_C_to_D_avg",
                    min=0, max=1, step=0.001, value=p_C_to_D_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- probability of dying in ICU [fraction]"),
                dbc.Input(
                    id="p_C_to_D_err", type="number", placeholder="p_C_to_D_err",
                    min=0, max=1, step=0.001, value=p_C_to_D_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average incubation time [days]"),
                dbc.Input(
                    id="T_incubation_avg", type="number", placeholder="T_incubation_avg",
                    min=1, max=21, step=0.05, value=T_incubation_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- incubation time [days]"),
                dbc.Input(
                    id="T_incubation_err", type="number", placeholder="T_incubation_err",
                    min=0, max=7, step=0.1, value=T_incubation_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average infectious time [days]"),
                dbc.Input(
                    id="T_infectious_avg", type="number", placeholder="T_infectious_avg",
                    min=1, max=21, step=0.05, value=T_infectious_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- infectious time [days]"),
                dbc.Input(
                    id="T_infectious_err", type="number", placeholder="T_infectious_err",
                    min=0, max=7, step=0.1, value=T_infectious_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average time from infectious to ICU [days]"),
                dbc.Input(
                    id="T_I_to_C_avg", type="number", placeholder="T_I_to_C_avg",
                    min=1, max=21, step=0.05, value=T_I_to_C_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- time from infectious to ICU [days]"),
                dbc.Input(
                    id="T_I_to_C_err", type="number", placeholder="T_I_to_C_err",
                    min=0, max=7, step=0.1, value=T_I_to_C_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average time to die in ICU [days]"),
                dbc.Input(
                    id="T_C_to_D_avg", type="number", placeholder="T_C_to_D_avg",
                    min=1, max=21, step=0.05, value=T_C_to_D_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- time  to die in ICU [days]"),
                dbc.Input(
                    id="T_C_to_D_err", type="number", placeholder="T_C_to_D_err",
                    min=0, max=7, step=0.1, value=T_C_to_D_err,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("average time to recover in ICU [days]"),
                dbc.Input(
                    id="T_C_to_R_avg", type="number", placeholder="T_C_to_R_avg",
                    min=1, max=21, step=0.05, value=T_C_to_R_avg,
                )
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("+/- time  to recover in ICU [days]"),
                dbc.Input(
                    id="T_C_to_R_err", type="number", placeholder="T_C_to_R_err",
                    min=0, max=7, step=0.1, value=T_C_to_R_err,
                )
            ]
        ),

        dbc.Button("Apply", id="submit-button-state",
                   color="primary", block=True)
    ],
    body=True,
)

# layout for the whole page
app.layout = dbc.Container(
    [
        # first, a jumbotron for the description and title
        dbc.Jumbotron(
            [
                dbc.Container(
                    [
                        html.H1("MonteCarlo", className="display-3"),
                        html.P(
                            "Graph results, later change into interactively graph... ",
                            className="lead",
                        ),
                        html.Hr(className="my-2"),
                        dcc.Markdown('''

                                        put some general instructions here 
                            '''
                                     )
                    ],
                    fluid=True,
                )
            ],
            fluid=True,
            className="jumbotron bg-white text-dark"
        ),
        # now onto the main page, i.e. the controls on the left
        # and the graphs on the right.
        dbc.Row(
            [
                # here we place the controls we just defined,
                # and tell them to use up the left 3/12ths of the page.
                dbc.Col(controls, md=3),
                # now we place the graphs on the page, taking up
                # the right 9/12ths.
                dbc.Col(
                    [
                        #
                        dcc.Graph(id='fig1'),
                        # the graph
                        dcc.Graph(id='fig2'),
                        # the next two graphs don't need as much space, so we
                        # put them next to each other in one row.
                        dbc.Row(
                            [
                                # the graph for the fatality rate over time.
                                dbc.Col(dcc.Graph(id='fig3'), md=6),
                                # the graph for the daily deaths over time.
                                dbc.Col(dcc.Graph(id="fig4"), md=6)

                            ]
                        ),
                    ],
                    md=9
                ),
            ],
            align="top",
        ),
    ],
    # fluid is set to true so that the page reacts nicely to different sizes etc.
    fluid=True,
)


#
# app.layout = html.Div(children=[
#     html.H1(children='SEIR Model, Monte Carlo results'),
#     html.H2(children=disease),
#     html.Div(children=['Population size =', N]),
#     html.Div(children=['initial number of Infected =', initial_cases]),
#     html.Div(children=['number of Monte Carlo Runs =', num_reps]),
#     html.Div(children='other important parameters, all with uniformly-distributed ranges set in the code:'
#                       ' % of immuned population, R0, T_incubation, T_infected, '
#                       'P(I2C), P(C2D), PT(I2C), T(C2R), T(C2D)'),
#     html.Div(children='''
#         I wish I could add an interactive section to present and adjust all parameters,
#         not only initial population...
#         and to find a way to arrange the figure side by side rather than one after the other...
#     '''),
#     dcc.Graph(
#         id='fig1',
#         figure=fig1,
#         className="six columns",
#     ),
#     dcc.Graph(
#         id='fig2',
#         figure=fig2,
#         className="six columns",
#     ),
#     dcc.Graph(
#         id='fig3',
#         figure=fig3,
#         className="six columns",
#     ),
#
#     dcc.Graph(
#         id='fig4',
#         figure=fig4,
#     ),
#     dcc.Graph(
#         id='fig5',
#         figure=fig5,
#     ),
#     dcc.Graph(
#         id='fig6',
#         figure=fig6,
#     ),
#     html.Div(children='Please take the median graphs below with a grain of salt,'
#                       ' as the median will reflect the mid-range of the parameters distribution, '
#                       'and to an extent will negate the uniform distribution we chose for the analysis'),
#     dcc.Graph(
#         id='fig10',
#         figure=fig10,
#     ),
#     dcc.Graph(
#         id='fig7',
#         figure=fig7,
#     ),
#     dcc.Graph(
#         id='fig8',
#         figure=fig8,
#     ),
#     dcc.Graph(
#         id='fig9',
#         figure=fig9,
#     )

###   keep the section below commented out to see what did work and what had not (which isdoubly commented out)
# dcc.Graph(
#     id='example-graph',
#     figure={
#         'data': [
#             # {'y': df_C_vs_t.T, 'type': 'line'}
#             {'y': df_C_vs_t.T[2], 'type': 'line'},
#             {'y': df_C_vs_t.T[3], 'type': 'line'}
#         ],
#         'layout': {
#             'title': disease
#         }
#     }
# )
# ])
# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'})

############################################ the dash app callbacks ################################################


@app.callback(
    [dash.dependencies.Output('fig1', 'figure'),
     dash.dependencies.Output('fig2', 'figure'),
     dash.dependencies.Output('fig3', 'figure'),
     dash.dependencies.Output('fig4', 'figure'),
     ],

    [dash.dependencies.Input('submit-button-state', 'n_clicks')],

    [dash.dependencies.State('num_reps', 'value'),
     dash.dependencies.State('initial_cases', 'value'),
     dash.dependencies.State('population', 'value'),
     dash.dependencies.State('R_0_avg', 'value'),
     dash.dependencies.State('R_0_err', 'value'),
     dash.dependencies.State('p_immune_avg', 'value'),
     dash.dependencies.State('p_immune_err', 'value'),
     dash.dependencies.State('p_I_to_C_avg', 'value'),
     dash.dependencies.State('p_I_to_C_err', 'value'),
     dash.dependencies.State('p_C_to_D_avg', 'value'),
     dash.dependencies.State('p_C_to_D_err', 'value'),
     dash.dependencies.State('T_incubation_avg', 'value'),
     dash.dependencies.State('T_incubation_err', 'value'),
     dash.dependencies.State('T_infectious_avg', 'value'),
     dash.dependencies.State('T_infectious_err', 'value'),
     dash.dependencies.State('T_I_to_C_avg', 'value'),
     dash.dependencies.State('T_I_to_C_err', 'value'),
     dash.dependencies.State('T_C_to_D_avg', 'value'),
     dash.dependencies.State('T_C_to_D_err', 'value'),
     dash.dependencies.State('T_C_to_R_avg', 'value'),
     dash.dependencies.State('T_C_to_R_err', 'value'),
     ]
)
def update_graph(_, num_reps, initial_cases, population, R_0_avg, R_0_err, p_immune_avg, p_immune_err, p_I_to_C_avg,
                 p_I_to_C_err, p_C_to_D_avg, p_C_to_D_err,
                 T_incubation_avg, T_incubation_err, T_infectious_avg, T_infectious_err, T_I_to_C_avg, T_I_to_C_err,
                 T_C_to_D_avg, T_C_to_D_err, T_C_to_R_avg, T_C_to_R_err):
    p_immune = np.random.uniform(p_immune_avg - p_immune_err, p_immune_avg + p_immune_err, num_reps).round(
        3)  # portion of immuned population
    R_0 = np.random.uniform(R_0_avg - R_0_err, R_0_avg + R_0_err, num_reps).round(2)  # Initial R0
    T_infectious = np.random.uniform(T_infectious_avg - T_infectious_err, T_infectious_avg + T_infectious_err,
                                     num_reps).round(2)  # infectious period [days]
    T_incubation = np.random.uniform(T_incubation_avg - T_incubation_err, T_incubation_avg + T_incubation_err,
                                     num_reps).round(2)  # incubation period [days] - not contagious during this time
    T_I_to_C = np.random.uniform(T_I_to_C_avg - T_I_to_C_err, T_I_to_C_avg + T_I_to_C_err, num_reps).round(
        2)  # time to transition from Infected to Critical [days]
    T_C_to_D = np.random.uniform(T_C_to_D_avg - T_C_to_D_err, T_C_to_D_avg + T_C_to_D_err, num_reps).round(
        2)  # time to transition from Critical to Dead [days]
    T_C_to_R = np.random.uniform(T_C_to_R_avg - T_C_to_R_err, T_C_to_R_avg + T_C_to_R_err, num_reps).round(
        2)  # time to transition from Critical to Recovered [days]
    p_I_to_C = np.random.uniform(p_I_to_C_avg - p_I_to_C_err, p_I_to_C_avg + p_I_to_C_err, num_reps).round(
        3)  # overall probability of transition from Infected to Critical
    p_C_to_D = np.random.uniform(p_C_to_D_avg - p_C_to_D_err, p_C_to_D_avg + p_C_to_D_err, num_reps).round(
        3)  # overall probability of transition from Infected to Critical

    gamma = 1.0 / T_infectious
    delta = 1.0 / T_incubation
    beta = R_0 / T_infectious
    initial_date = 0

    df_results = pd.DataFrame({})  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
    df_I_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
    I_vs_t = []
    df_C_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
    C_vs_t = []
    df_D_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
    D_vs_t = []
    df_IFR_vs_t = pd.DataFrame([])  # initiate as an empty DataFrame so I can append to it in the Monte Carlo loop
    IFR_vs_t = []
    for i in range(num_reps):
        time, S, E, I, C, R, D, total_CFR, daily_CFR = Model(initial_cases, initial_date, population, p_immune[i],
                                                             beta[i],
                                                             gamma[i],
                                                             delta[i], p_I_to_C[i], T_I_to_C[i], p_C_to_D[i],
                                                             T_C_to_R[i],
                                                             T_C_to_D[i])
        df_temp = pd.DataFrame(index=[i], data={'R_0': R_0[i], 'immune %': p_immune[i] * 100,
                                                'T_infectious': T_infectious[i], 'T_incubation': T_incubation[i],
                                                'p_I_to_C%': p_I_to_C[i] * 100, 'p_C_to_D%': p_C_to_D[i] * 100,
                                                'T_I_to_C': T_I_to_C[i], 'T_C_to_R': T_C_to_R[i],
                                                'T_C_to_D': T_C_to_D[i],
                                                'peak E': max(E), 'peak I': max(I), 'peak R': max(R),
                                                'peak C': max(C), 'max D': max(D), 'total IFR': max(total_CFR)})
        df_results = df_results.append(df_temp)
        # plot progress to terminal, to avoid the feeling it got stuck...
        if round(i / 25, 0) == (i / 25):
            print('in MC round ', i, 'out of ', num_reps)

        # thanks to
        # https://stackoverflow.com/questions/31674557/how-to-append-rows-in-a-pandas-dataframe-in-a-for-loop for this method
        IFR_vs_t.append(total_CFR)
        I_vs_t.append(I)
        D_vs_t.append(D)
        C_vs_t.append(C)
        df_IFR_vs_t = pd.DataFrame(IFR_vs_t)  # compartment value for each day is appended as new row
        df_I_vs_t = pd.DataFrame(I_vs_t)  # compartment value for each day is appended as new row
        df_C_vs_t = pd.DataFrame(C_vs_t)  # compartment value for each day is appended as new row
        df_D_vs_t = pd.DataFrame(D_vs_t)  # compartment value for each day is appended as new row

    return {
        'data': [{'y': df_I_vs_t[1:20].T, 'type': 'line', 'name': 'fig1'}, ],
        'layout': {'title': 'fake title', }
        }, {
        'data': [{'y': df_I_vs_t[1:20].T, 'type': 'line', 'name': 'fig2'}, ],
        'layout': {'title': 'fake title', }
        }, {
        'data': [{'y': df_I_vs_t[1:20].T, 'type': 'line', 'name': 'fig3'}, ],
        'layout': {'title': 'fake title', }
        }, {
        'data': [{'y': df_I_vs_t[1:20].T, 'type': 'line', 'name': 'fig4'}, ],
        'layout': {'title': 'fake title', }
        }

if __name__ == '__main__':
    app.run_server(debug=True)
