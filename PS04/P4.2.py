
# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Problem Set #4 - Problem #2 : Ligand, Receptor, Complex Modelling
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Function for the changing L, R, C concetrations using a Simple ODE System 
def Simple_LRC(t, U, k1, k2): 
    R1, C1, C2, C3, L, X = U
    dR1_dt = (k2 * C1) + (k2 * C2) - (k1 * R1 * L) - (k1 * R1 * X)
    dC1_dt = (k1 * R1 * L) + (k2 * C3) - (k2 * C1) - (k1 * C1 * X)
    dC2_dt = (k1 * R1 * X) + (k2 * C3) - (k2 * C2) - (k1 * C2 * L) 
    dC3_dt = (k1 * C2 * L) + (k1 * C1 * X) - (k2 * C3)
    dL_dt = (k2 * C1) + (k2 * C3) - (k1 * R1 * L) - (k1 * C2 * L)
    dX_dt = (k2 * C2) + (k2 * C3) - (k1 * X * R1) - (k1 * X * C1)
    return [dR1_dt, dC1_dt, dC2_dt, dC3_dt, dL_dt, dX_dt]

# Function for solving and plotting the solution to the Simple LRC ODE System
def LRC_SimpleSolution(k1, k2, R0, L0, X0, Condition, time, time_span):
    C10, C20, C30 = 0, 0, 0 # Initial values for the complexes
    initial = [R0, C10, C20, C30, L0, X0]

    # Solve the Ligand, Receptor, Complex ODE System using Stiff Solver for faster 
    LRCSol = solve_ivp(Simple_LRC, time, initial, args = (k1, k2), t_eval = time_span, method = 'Radau')

    # Plot the vlaues for the three complexes
    plt.plot(LRCSol.t, LRCSol.y[1], label = 'Complex [C1]', color = 'darkblue')
    plt.plot(LRCSol.t, LRCSol.y[2], label = 'Complex [C2]', color = 'goldenrod')
    plt.plot(LRCSol.t, LRCSol.y[3], label = 'Complex [C3]', color = 'red')

    plt.xlabel('Time [s]')
    plt.ylabel('Complex Concetration [#/cell]')
    plt.title('Simple: ' + Condition)
    plt.legend()
    plt.grid(True)
    plt.show()
    return 

# Define time span
time = (0, 1) 
time_span = np.linspace(0, 1, 100)

# Using Interferon and Human Interferon a2a in A549 Values

# EXPIREMENT [1] : Equal Ligand and Adaptor Concentrations 
Condition_1 = ('Equal Ligand and Adaptor Concentrations')
R0, k1, k2, Kd = 900, 2e8, 0.072, 3.3e-10
L0, X0 = 500, 500
LRC_SimpleSolution(k1, k2, R0, L0, X0, Condition_1, time, time_span)

# EXPIREMENT [2] : High Ligand and Low Adaptor Concetration 
Condition_2 = ('High Ligand and Low Adaptor Concetration')
L0, X0 = 2000, 500 
LRC_SimpleSolution(k1, k2, R0, L0, X0, Condition_2, time , time_span)

# EXPIREMENT [3] : Low Ligand and High Adaptor Concetration 
Condition_3 = ('Low Ligand and High Adaptor Concetration')
L0, X0 = 500, 2000
LRC_SimpleSolution(k1, k2, R0, L0, X0, Condition_3, time , time_span)

# -----------------------------------------------------------------------------------------------------------------------------------
# PART [B] : Reaction / topological change to system 

# Function for the changing L, R, C concetrations using a more Complex ODE System w/ additional rate constants
def Complex_LRC(t, U, k1C1, k2C1, k1C2, k2C2, k1C3, k2C3):
    R1, C1, C2, C3, L, X = U
    dR1_dt = (k2C1 * C1) + (k2C2 * C2) - (k1C1 * R1 * L) - (k1C2 * R1 * X)
    dC1_dt = (k1C1 * R1 * L) + (k2C3 * C3) - (k2C1 * C1) - (k1C3 * C1 * X)
    dC2_dt = (k1C2 * R1 * X) + (k2C3 * C3) - (k2C2 * C2) - (k1C1 * C2 * L) 
    dC3_dt = (k1C3 * C1 * X) + (k1C1 * C2 * L) - (k2C3 * C3)
    dL_dt = (k2C1 * C1) + (k2C3 * C3) - (k1C1 * R1 * L) - (k1C1 * C2 * L)
    dX_dt = (k2C2 * C2) + (k2C3 * C3) - (k1C2 * X * R1) - (k1C3 * X * C1)
    return [dR1_dt, dC1_dt, dC2_dt, dC3_dt, dL_dt, dX_dt]

# Function for solving and plotting the solution to the Complex LRC ODE System
def LRC_ComplexSolution(k1C1, k2C1, k1C2, k2C2, k1C3, k2C3, R0, L0, X0, Condition, time, time_span): 
    C10, C20, C30 = 0, 0, 0 # Initial values for the complexes
    initial = [R0, C10, C20, C30, L0, X0]
 
    # Solve the Ligand, Receptor, Complex ODE System using Stiff Solver for faster 
    LRCSol = solve_ivp(Complex_LRC, time, initial, args = (k1C1, k2C1, k1C2, k2C2, k1C3, k2C3), t_eval = time_span, method = 'Radau')

    # Plot the vlaues for the three complexes
    plt.plot(LRCSol.t, LRCSol.y[1], label = 'Complex [C1]', color = 'darkblue')
    plt.plot(LRCSol.t, LRCSol.y[2], label = 'Complex [C2]', color = 'goldenrod')
    plt.plot(LRCSol.t, LRCSol.y[3], label = 'Complex [C3]', color = 'red')

    plt.xlabel('Time [s]')
    plt.ylabel('Complex Concetration [#/cell]')
    plt.title('Complex: ' + Condition)
    plt.legend()
    plt.grid(True)
    plt.show()
    return 

# Repeat Experiements with the new complex system 
def Perturbations(k_constants):
    k1C1, k2C1, k1C2, k2C2, k1C3, k2C3 = k_constants

    L0, X0 = 500, 500
    LRC_ComplexSolution(k1C1, k2C1, k1C2, k2C2, k1C3, k2C3, R0, L0, X0, Condition_1, time, time_span)

    L0, X0 = 2000, 500 
    LRC_ComplexSolution(k1C1, k2C1, k1C2, k2C2, k1C3, k2C3, R0, L0, X0, Condition_2, time , time_span)

    L0, X0 = 500, 2000
    LRC_ComplexSolution(k1C1, k2C1, k1C2, k2C2, k1C3, k2C3, R0, L0, X0, Condition_3, time , time_span)
    return 

# EXPERIMENT [1] : Close to true (k1, and k2) values 
k_constants = [2.1e8, 0.075, 1.9e8, 0.07, 1.5e8, 0.08]
Perturbations(k_constants)

# EXPERIMENT [2] : Complexes are relatively unstable 
k_constants = [1e7, 0.5, 8e6, 0.6, 5e6, 0.7]
Perturbations(k_constants)

# EXPERIMENT [3] : Balanced set of parameters with high association and high dissociation
k_constants = [5e7, 0.2, 4e7, 0.25, 3e7, 0.3]
Perturbations(k_constants)