import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html


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


N = 100000  # size of initial population
initial_cases = 10  # initial number of Exposed. I chose 10 to stay away from the critical community size at the onset of the epidemic.

num_reps = 1000  # for Monte Carlo

disease = 'Covid_19'  # supported options: 'Covid_19', 'Influenza_seasonal', or 'deterministic_test'
print(disease)
# times below are assumed to be common for Covid-19 and influenza seasonal
# todo: if supporting more disease, and they don't share the same timing, move those into the "if disease " above.
T_I_to_C = np.random.uniform(4.5, 12.5, num_reps).round(2)  # time to transition from Infected to Critical [days]
T_C_to_D = np.random.uniform(5, 9, num_reps).round(2)  # time to transition from Critical to Dead [days]
T_C_to_R = np.random.uniform(12, 16, num_reps).round(2)  # time to transition from Critical to Recovered [days]

# The section below is intended to be modular, so we can easily add other disease model.
if disease == 'Covid_19':
    R_0 = np.random.uniform(1.94, 5.7, num_reps).round(2)  # Initial R0
    p_immune = np.random.uniform(5 / 100, 7 / 100, num_reps).round(3)  # portion of immuned population
    T_infectious = np.random.uniform(8, 12, num_reps).round(2)  # infectious period [days]
    T_incubation = np.random.uniform(3, 8, num_reps).round(
        2)  # incubation period [days] - not contagious during this time
    gamma = 1.0 / T_infectious
    delta = 1.0 / T_incubation
    beta = R_0 / T_infectious
    p_I_to_C = np.random.uniform(3 / 100, 8 / 100, num_reps).round(
        3)  # overall probability of transition from Infected to Critical
    p_C_to_D = np.random.uniform(25 / 100, 40 / 100, num_reps).round(
        3)  # overall probability of transition from Infected to Critical
elif disease == 'Influenza_seasonal':
    R_0 = np.random.uniform(0.9, 2.1, num_reps).round(2)  # Initial R0
    p_immune = np.random.uniform(15 / 100, 25 / 100, num_reps).round(3)  # portion of immuned population
    T_infectious = np.random.uniform(3, 10, num_reps).round(2)  # infectious period [days]
    T_incubation = np.random.uniform(1, 4, num_reps).round(
        2)  # incubation period [days] - not contagious during this time
    gamma = 1.0 / T_infectious
    delta = 1.0 / T_incubation
    beta = R_0 / T_infectious
    p_I_to_C = np.random.uniform(0.25 / 100, 0.75 / 100, num_reps).round(
        3)  # overall probability of transition from Infected to Critical
    p_C_to_D = np.random.uniform(33 / 100, 48 / 100, num_reps).round(
        3)  # overall probability of transition from Infected to Critical
elif disease == 'deterministic_test':
    R_0 = np.random.uniform(3, 3, num_reps).round(2)  # Initial R0
    p_immune = np.random.uniform(0 / 100, 0 / 100, num_reps).round(3)  # portion of immuned population
    T_infectious = np.random.uniform(9, 9, num_reps).round(2)  # infectious period [days]
    T_incubation = np.random.uniform(3, 3, num_reps).round(
        2)  # incubation period [days] - not contagious during this time
    gamma = 1.0 / T_infectious
    delta = 1.0 / T_incubation
    beta = R_0 / T_infectious
    p_I_to_C = np.random.uniform(5 / 100, 5 / 100, num_reps).round(
        3)  # overall probability of transition from Infected to Critical
    p_C_to_D = np.random.uniform(30 / 100, 30 / 100, num_reps).round(
        3)  # overall probability of transition from Infected to Critical
    T_I_to_C = np.random.uniform(12, 12, num_reps).round(2)  # time to transition from Infected to Critical [days]
    T_C_to_D = np.random.uniform(7.5, 7.5, num_reps).round(2)  # time to transition from Critical to Dead [days]
    T_C_to_R = np.random.uniform(6.5, 6.5, num_reps).round(2)  # time to transition from Critical to Recovered [days]
else:
    print('undefined disease!')

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
    time, S, E, I, C, R, D, total_CFR, daily_CFR = Model(initial_cases, initial_date, N, p_immune[i], beta[i], gamma[i],
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
    if round(i/25, 0) == (i/25):
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
# but actually print to terminal only if <30 rows (30 being arbitrary, used for testing)
if num_reps < 30:
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


app = dash.Dash()
#  plot using ploty express, with the intention to transfer it to DCC as explained in https://dash.plotly.com/dash-core-components/graph
#  could not make it run with the regular dcc.Graph( ...  id='example-graph', ... figure={ ...  'data': [
#  in that case it worked for a single line but I couldn't figure out how to pull all lines from the DataFrame. It kept failing to load the web page.
# subplots: by https://panel.holoviz.org/reference/panes/Plotly.html
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

fig7 = px.line(df_I_vs_t.T.apply(np.median, axis=1),
               title="Median of Infected vs. time, over all Monte Carlo run",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Median of Infected"
               },
               )
fig7.layout.update(showlegend=False)  # eliminate legends
fig8 = px.line(df_C_vs_t.T.apply(np.median, axis=1),
               title="Median of Critical vs. time, over all Monte Carlo run",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Median of Critical"
               },
               )
fig8.layout.update(showlegend=False)  # eliminate legends

fig9 = px.line(df_D_vs_t.T.apply(np.median, axis=1),
               title="Median of Dead vs. time, over all Monte Carlo run",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Median of Dead"
               },
               )
fig9.layout.update(showlegend=False)  # eliminate legends

fig10 = px.line(df_IFR_vs_t.T.apply(np.median, axis=1),
               title="Median of IFR vs. time, over all Monte Carlo run",
               width=600, height=400,
               labels={  # replaces default labels by column name
                   # ... thanks to https://plotly.com/python/styling-plotly-express/
                   "index": "time [days]", "value": "Median of IFR"
               },
               )
fig10.layout.update(showlegend=False)  # eliminate legends



app.layout = html.Div(children=[
    html.H1(children='SEIR Model, Monte Carlo results'),
    html.H2(children=disease),
    html.Div(children=['Population size =', N]),
    html.Div(children=['initial number of Infected =', initial_cases]),
    html.Div(children=['number of Monte Carlo Runs =', num_reps]),
    html.Div(children='other important parameters, all with uniformly-distributed ranges set in the code:'
                      ' % of immuned population, R0, T_incubation, T_infected, '
                      'P(I2C), P(C2D), PT(I2C), T(C2R), T(C2D)'),
    html.Div(children='''
        I wish I could add an interactive section to present and adjust all parameters, 
        not only initial population... 
        and to find a way to arrange the figure side by side rather than one after the other...  
    '''),
    dcc.Graph(
        id='fig1',
        figure=fig1,
    ),
    dcc.Graph(
        id='fig2',
        figure=fig2,
    ),
    dcc.Graph(
        id='fig3',
        figure=fig3,
    ),

    dcc.Graph(
        id='fig4',
        figure=fig4,
    ),
    dcc.Graph(
        id='fig5',
        figure=fig5,
    ),
    dcc.Graph(
        id='fig6',
        figure=fig6,
     ),
    html.Div(children='Please take the median graphs below with a grain of salt,'
                      ' as the median will reflect the mid-range of the parameters distribution, '
                      'and to an extent will negate the uniform distribution we chose for the analysis'),
    dcc.Graph(
        id='fig10',
        figure=fig10,
    ),
    dcc.Graph(
        id='fig7',
        figure=fig7,
    ),
    dcc.Graph(
        id='fig8',
        figure=fig8,
    ),
    dcc.Graph(
        id='fig9',
        figure=fig9,
    )

    ###   keep the section below commented out to see what did work and what had not (which id doubly commented out)
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
])

if __name__ == '__main__':
    app.run_server(debug=True)
