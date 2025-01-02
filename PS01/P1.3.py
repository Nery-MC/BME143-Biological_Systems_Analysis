
# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Problem Set #1 - Problem #3 : Protein Phosphorylation Dynamics

from scipy.integrate import solve_ivp
import numpy as np 
import matplotlib.pyplot as plt
import random

# PART [A] : set rate constant (r) and protein concentration 
r01 = 0.5 # [time^-1] State P0 to P1 (Phosphorylation)
r12 = 0.5 # [time^-1] State P1 to P2 (Phosphorylation)
r21 = 0.5 # [time^-1] State P2 to P1 (Dephosphorylation)
r10 = 0.5 # [time^-1] State P1 to P0 (Dephosphorylation)

P0i = 1 # [molecules/cell] Initial concentration in the P0 State 
P1i = 1 # [molecules/cell] Initial Concentration in the P1 State 
P2i = 1 # [molecules/cell] Initial concetration in the P2 State 

# ODE system for the changing concentration of each phosphorylation state
def Phosphorylation(t, U, r01, r12, r21, r10): 
    P0, P1, P2 = U 
    # a negative rate constant reflects molecules leaving that state (ie, dephosphorylating)
    dP0_dt = (r10 * P1) + (-r01 * P0) 
    dP1_dt = (r01 * P0) + (r21 * P2) + (-r12 * P1) + (-r10 * P1)
    dP2_dt = (r12 * P1) + (-r21 * P2)
    return (dP0_dt, dP1_dt, dP2_dt)

# Time span and initial conditions 
time = (0, 100) 
initial = (P0i, P1i, P2i)

# Solution to the phosphorylaton ODE system 
solution = solve_ivp(Phosphorylation, time, initial, args = (r01, r12, r21, r10))

# Plot values 
plt.plot(solution.t, solution.y[0], label = 'P0 [Unphosphorylated]')
plt.plot(solution.t, solution.y[1], label = 'P1 [Single Phosphorylation]')
plt.plot(solution.t, solution.y[2], label = 'P2 [Double Phosphorylation]')

plt.xlabel('t [Time]')
plt.ylabel('Protein Concentration')
plt.title('Protein Phosphorylation : [P]i = 1 and [r]i = 0.5')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------------------------------
# PART [B] : different sets of initial conditions for P0, P1, and P2 
initial = np.array([1, 2, 3]) # Update initial concentrations into an array for shuffling
j = 0 # Counter to make sure its 6 iterations

# Iterate until you have generated 6 graphs 
while (j < 6): 
    random.shuffle(initial) # Randomizes/shuffles the values in 'initial'
    P0i, P1i, P2i = initial # Unpack shuffled initial vlaues
    solution = solve_ivp(Phosphorylation, time, initial, args = (r01, r12, r21, r10)) 

    # Plot the random initial vlaues 
    plt.plot(solution.t, solution.y[0], label = 'P0 [Unphosphorylated]')
    plt.plot(solution.t, solution.y[1], label = 'P1 [Single Phosphorylation]')
    plt.plot(solution.t, solution.y[2], label = 'P2 [Double Phosphorylation]')

    plt.title(f'Initial : [P0] = {P0i}, [P1] = {P1i}, [P2] = {P2i}')
    plt.xlabel('t [Time]')
    plt.ylabel('Protein Concentration')
    plt.grid(True)
    plt.show()
    
    # Update the counter 
    j = j + 1

# --------------------------------------------------------------------------
# PART [C] : different sets of rate contants values 
initial = (1, 1, 1) # Set it back to original
r01, r12, r21, r10 = (0.5, 0.25, 0.10, 0.30) 
rate = np.array([r01, r12, r21, r10]) # Rate array that hold constants 

j = 0 # Counter to make sure its 6 iterations

# Iterate until you have generated 6 graphs 
while (j < 6): 
    random.shuffle(rate) # Randomizes/shuffles the values in ea 'rate' constant
    r01, r12, r21, r10 = rate # Unpack shuffled vlaues 
    solution = solve_ivp(Phosphorylation, time, initial, args = (r01, r12, r21, r10)) 

    # Plot the random initial vlaues 
    plt.plot(solution.t, solution.y[0], label = 'P0 [Unphosphorylated]')
    plt.plot(solution.t, solution.y[1], label = 'P1 [Single Phosphorylation]')
    plt.plot(solution.t, solution.y[2], label = 'P2 [Double Phosphorylation]')

    plt.title(f'Rates : [r01] = {r01}, [r12] = {r12}, [r21] = {r21}, [r10] = {r10}')
    plt.xlabel('t [Time]')
    plt.ylabel('Protein Concentration')
    plt.grid(True)
    plt.show()
    
    # Update the counter 
    j = j + 1

    # --------------------------------------------------------------------------
# PART [D] : different sets of initial and rate-contants values 
initial = np.array([1, 2, 3]) # Initial array
r01, r12, r21, r10 = (0.5, 0.25, 0.10, 0.30) 
rate = np.array([r01, r12, r21, r10]) # Rate array

j = 0 # Counter 
while (j < 6): 
    random.shuffle(rate)
    r01, r12, r21, r10 = rate # Unpack the shuffled rate constant
    random.shuffle(initial)
    P0i, P1i, P2i = initial # Unpack the shuffled initial vlaues 

    solution = solve_ivp(Phosphorylation, time, initial, args = (r01, r12, r21, r10)) 

    # Plot the random initial vlaues 
    plt.plot(solution.t, solution.y[0], label = 'P0 [Unphosphorylated]')
    plt.plot(solution.t, solution.y[1], label = 'P1 [Single Phosphorylation]')
    plt.plot(solution.t, solution.y[2], label = 'P2 [Double Phosphorylation]')

    plt.title(f'Rates : [r01] = {r01}, [r12] = {r12}, [r21] = {r21}, [r10] = {r10}\n' 
              f'[P0] = {P0i}, [P1] = {P1i}, [P2] = {P2i}')
    plt.xlabel('t [Time]')
    plt.ylabel('Protein Concentration')
    plt.grid(True)
    plt.show()
    j = j + 1