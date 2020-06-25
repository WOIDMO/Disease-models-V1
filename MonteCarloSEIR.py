import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.integrate import odeint
import matplotlib.pyplot as plt

############################################ the model ################################################
def deriv(y, t, N, p_immune, beta, gamma, delta, p_I_to_C, T_I_to_C, p_C_to_D, T_C_to_R, T_C_to_D):
    S, E, I, C, R, D = y
    dSdt = -beta * I * S * (1-p_immune) / N
    dEdt = beta * I * S * (1-p_immune) / N - delta * E
    dIdt = delta * E - 1/T_I_to_C * p_I_to_C * I - gamma * (1 - p_I_to_C) * I
    dCdt = 1/T_I_to_C * p_I_to_C * I - 1/T_C_to_D * p_C_to_D * C -  (1 - p_C_to_D) * 1/T_C_to_R * C
    dRdt = gamma * (1 - p_I_to_C) * I + (1 - p_C_to_D) * 1/T_C_to_R * C
    dDdt = 1/T_C_to_D * p_C_to_D * C
    return dSdt, dEdt, dIdt, dCdt, dRdt, dDdt

def Model(initial_cases, initial_date, N, p_immune, beta, gamma, delta, p_I_to_C, T_I_to_C, p_C_to_D, T_C_to_R, T_C_to_D):
    days = 365
    y0 = N-initial_cases, initial_cases, 0.0, 0.0, 0.0, 0.0
    t = np.linspace(1, days, days) # ORN changed start = 0 to start = 1 for integer values
#    print(t)
    ret = odeint(deriv, y0, t, args=(N, p_immune, beta, gamma, delta, p_I_to_C, T_I_to_C, p_C_to_D, T_C_to_R, T_C_to_D))
    #todo: make sure to find a way to implement the critical community size truncating - Actually, Steve said no need
    S, E, I, C, R, D = ret.T
    # note this is actually IFR: Dead / total_infected
    total_CFR = [0] + [100 * D[i] / sum(delta*E[:i]) if sum(
        delta*E[:i]) > 0 else 0 for i in range(1, len(t))]
    daily_CFR = [0] + [100 * ((D[i]-D[i-1]) / ((R[i]-R[i-1]) + (D[i]-D[i-1]))) if max(
        (R[i]-R[i-1]), (D[i]-D[i-1])) > 10 else 0 for i in range(1, len(t))]
    return t, S, E, I, C, R, D, total_CFR, daily_CFR

############### initialization ###############


N = 100000  # size of initial population
initial_cases = 10  # initial number of Exposed. I chose 10 to stay away from the critical community size at the onset of the epidemic.

num_reps = 1000  # for Monte Carlo

disease = 'Influenza_seasonal'  # supported options: 'Covid_19', 'Influenza_seasonal'
print(disease)

# The section below is intended to be modular, so we can easily add other disease model.
if disease == 'Covid_19':
    R_0 = np.random.uniform(1.94, 5.7, num_reps).round(2)  # Initial R0
    p_immune = np.random.uniform(5/100, 7/100, num_reps).round(3)  # portion of immuned population
    T_infectious = np.random.uniform(8, 12, num_reps).round(2)  # infectious period [days]
    T_incubation = np.random.uniform(3, 8, num_reps).round(2)  # incubation period [days] - not contagious during this time
    gamma = 1.0 / T_infectious
    delta = 1.0 / T_incubation
    beta = R_0 / T_infectious
    p_I_to_C = np.random.uniform(3/100, 8/100, num_reps).round(3) # overall probability of transition from Infected to Critical
    p_C_to_D = np.random.uniform(25/100, 40/100, num_reps).round(3) # overall probability of transition from Infected to Critical
elif disease == 'Influenza_seasonal':
    R_0 = np.random.uniform(0.9, 2.1, num_reps).round(2)  # Initial R0
    p_immune = np.random.uniform(15/100, 25/100, num_reps).round(3)  # portion of immuned population
    T_infectious = np.random.uniform(3, 10, num_reps).round(2)  # infectious period [days]
    T_incubation = np.random.uniform(1, 4, num_reps).round(2)  # incubation period [days] - not contagious during this time
    gamma = 1.0 / T_infectious
    delta = 1.0 / T_incubation
    beta = R_0 / T_infectious
    p_I_to_C = np.random.uniform(0.25/100, 0.75/100, num_reps).round(3) # overall probability of transition from Infected to Critical
    p_C_to_D = np.random.uniform(33/100, 48/100, num_reps).round(3) # overall probability of transition from Infected to Critical
else:
    print('undefined disease!')


# times below are assumed to be common for Covid-19 and influenza seasonal
# todo: if supporting more disease, and they don't share the same timing, move those into the "if disease " above.
T_I_to_C = np.random.uniform(4.5, 12.5, num_reps).round(2) # time to transition from Infected to Critical [days]
T_C_to_D = np.random.uniform(5, 9, num_reps).round(2) # time to transition from Critical to Dead [days]
T_C_to_R = np.random.uniform(12, 16, num_reps).round(2) # time to transition from Critical to Recovered [days]
initial_date = 0

############### Monte Carlo ###############
print('start Monte-Carlo')
results_max_C = []
results_max_D = []
results_tot_CFR = []
for i in range(num_reps):
    time, S, E, I, C, R, D, total_CFR, daily_CFR = Model(initial_cases, initial_date, N, p_immune[i], beta[i], gamma[i], delta[i],  p_I_to_C[i], T_I_to_C[i], p_C_to_D[i], T_C_to_R[i], T_C_to_D[i])
    results_max_C.append(max(C))
    results_max_D.append(max(D))
    results_tot_CFR.append(max(total_CFR))

    # todo: have to pull the plotting outside of the loop.
    #  Save results vectors and plot as one, outside the Monte Carlo loop.
    #  When doing that, plot peak Critical, total Dead and CFR (IFR) over time
    # plot Critical as a function of time
    plt.plot(time, C)
    plt.title('Critical compartment vs time, for all Monte Carlo runs')
    plt.xlabel('time [days]')
    plt.ylabel('Critical')
    plt.show()
    # # plot CFR as a function of time
    # plt.plot(time, total_CFR)
    # plt.title('total_CFR vs time, for all Monte Carlo runs')
    # plt.xlabel('time [days]')
    # plt.ylabel('total_CFR')
    # plt.show()

####################### collect data ##########################
print('re-arranging data')
df = pd.DataFrame(index=range(num_reps), data={'R_0': R_0, 'immune %': p_immune * 100,
                                               'T_infectious': T_infectious, 'T_incubation': T_incubation,
                                               'p_I_to_C%': p_I_to_C * 100, 'p_C_to_D%': p_C_to_D * 100,
                                               'T_I_to_C': T_I_to_C, 'T_C_to_R': T_C_to_R, 'T_C_to_D': T_C_to_D,
                                               'peak C': results_max_C, 'peak D': results_max_D,
                                               'total_CFR': results_tot_CFR})
# Set it to None to display all columns in the dataframe,
# but actually print to terminal only if <30 rows (30 being arbitrary, used for testing)
if num_reps < 30:
    pd.set_option('display.max_columns', None)
    print(df)
else:
    print('NOT printing df as it is deemed too long')

print('min / max C')
print(min(results_max_C), max(results_max_C))
print('min / max D')
print(min(results_max_D), max(results_max_D))
print('min(CFR), max(CFR)')
print(min(results_tot_CFR), max(results_tot_CFR))

####################### plot ##########################

print('now plot histogram')

df.hist(column='peak C')
plt.show()

df.hist(column='peak D')
plt.show()

df.hist(column='total_CFR')
plt.show()




