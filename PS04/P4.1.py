
# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Problem Set #4 - Problem #1 : Briggs-Haldane Approximation 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define constants 
kcat = 25 # [s^-1] catalytic constant
Km = 0.005 # [M] Michaelis constant
k1 = 1.0e6 # [M^-1 * s^-1] association constant

# Define initial condition value table
Initial_Conditions_mM = [(0.05, 0.05), (0.05, 0.005), (0.5, 0.5), (0.5, 0.05), (5, 5), 
                      (5, 0.5), (5, 0.05), (50, 5), (50, 0.5)] # [S0] and [E0] pair

# Convert each mM vlaue into a M vlaue 
Initial_Conditions_M = [((i[0] / 1000), (i[1] / 1000)) for i in Initial_Conditions_mM]

# Calculate kmin1 using eq Km = (kmin1 + kcat) / k1
kmin1 = (k1 * Km) - kcat
time = (0, 1)
time_span = np.linspace(0, 1, 100)

# Function for the changing concetrations using ODE System
def EnzymeKinetics(t, U, k1, kmin1, kcat):
    P, S, E, ES = U
    dP_dt = kcat * ES
    dS_dt = (-k1 * E * S) + (kmin1 * ES)
    dE_dt = (kmin1 * ES) - (k1 * E * S) + (kcat * ES)
    dES_dt = (k1 * E * S) - (kmin1 * ES) - (kcat * ES)
    return [dP_dt, dS_dt, dE_dt, dES_dt]

# Function for the Briggs-Haldane Approximation 
def BH(E0, S0, Km): 
    ES = (E0 * S0) / (Km + S0)
    return ES

# Solve the ODE System and Steady State System 
for i, (S0, E0) in enumerate(Initial_Conditions_M): 

    # Initialize initial conditions 
    ES0, P0 = 0, 0 # initial conditions for ES and P
    initial = [P0, S0, E0, ES0]

    # Solve the Enzyme Kinetics ODE System
    EnzymeSol = solve_ivp(EnzymeKinetics, time, initial, args = (k1, kmin1, kcat), t_eval = time_span)
    P_ODE = EnzymeSol.y[0] # Extract product concentration from ODE system

    # Plot the product concentration for ODE solution 
    plt.plot(EnzymeSol.t, P_ODE, label = 'ODE System', color = 'darkblue')

    # Solve the Briggs-Haldane Approx. using Psuedo Steady-State and find [P]
    ES_PSS = BH(E0, S0, Km) 
    P_PSS = kcat * ES_PSS * time_span

    # Plot the product concentration for the PSS solution
    plt.plot(time_span, P_PSS, label = 'BH-PSS System', color = 'darkcyan')

    # Calculate residuals 
    residuals = P_ODE - P_PSS

    # Plot the residuals on the same graph 
    plt.plot(time_span, residuals, label = 'Residual', color = 'red')

    # Finish off the graph
    plt.xlabel('Time [s]')
    plt.ylabel('Product Concetration [M]')
    plt.title(f'Product Concetration over Time for [S0 = {S0*1000} mM, E0 = {E0*1000} mM]')
    plt.legend()
    plt.grid(True)
    plt.show()















