# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Problem Set #5 - Problem #1 : Ferrel Huang MAP Kinase Cascade

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from f_huang import f

# Define the initial parameters
E1_Concentrations = np.logspace(-6, 1, 100)
Time_Span = [0, 100]
Y0 = [0.1, 0, 0.003, 0, 0, 0, 0.0003, 0, 1.2, 0, 0, 0, 0, 0.0003, 0, 0.0003, 
      0, 1.2, 0, 0, 0.12, 0, 0.12, 0]

# Function to extract the steady state of each kinase from solving the ODE 
def Kinase_SteadyState(E1_concentration):
    Y0[0] = E1_concentration
    Solution = solve_ivp(f, Time_Span, Y0, method='BDF')
    SS_MAPKKK, SS_MAPKK, SS_MAPK = Solution.y[3, -1], Solution.y[10, -1], Solution.y[19, -1]
    return SS_MAPKKK, SS_MAPKK, SS_MAPK

# Function for the Hill Function
def Hill(x, K, nH): 
    return (x**nH) / (K + (x**nH))

# Calculate steady-state values for a range of E1 concentrations
E1Total_Values = []
ODE_MAPKKK_Values, ODE_MAPKK_Values, ODE_MAPK_Values = [], [], []
HILL_MAPKKK_Values, HILL_MAPKK_Values, HILL_MAPK_Values = [], [], []

for E1 in E1_Concentrations:
    SS_MAPKKK, SS_MAPKK, SS_MAPK = Kinase_SteadyState(E1)
    E1Total_Values.append(E1)
    ODE_MAPKKK_Values.append(SS_MAPKKK)
    ODE_MAPKK_Values.append(SS_MAPKK)
    ODE_MAPK_Values.append(SS_MAPK)

    HILL_MAPKKK_Values.append(Hill(E1, 0.3, 1))
    HILL_MAPKK_Values.append(Hill(E1, 0.3, 1.7))
    HILL_MAPK_Values.append(Hill(E1, 0.3, 4.9))

# Determine EC100 for each kinase of each system 
EC100_MAPKKK_ODE = max(ODE_MAPKKK_Values)
EC100_MAPKK_ODE = max(ODE_MAPKK_Values)
EC100_MAPK_ODE = max(ODE_MAPK_Values)

EC100_MAPKKK_HILL = max(HILL_MAPKKK_Values)
EC100_MAPKK_HILL = max(HILL_MAPKK_Values)
EC100_MAPK_HILL = max(HILL_MAPK_Values)


# Function to calculate EC50 using interpolation
def EC50(E1_Values, Kinase_Value, EC100): 
    Interpolation = interp1d(Kinase_Value, E1_Values, kind = 'linear', fill_value = "extrapolate")
    return Interpolation(EC100 / 2)

# Calculate the EC50 for each Kinase
EC50_MAPKKK_ODE = EC50(E1Total_Values, ODE_MAPKKK_Values, EC100_MAPKKK_ODE)
EC50_MAPKK_ODE = EC50(E1Total_Values, ODE_MAPKK_Values, EC100_MAPKK_ODE)
EC50_MAPK_ODE = EC50(E1Total_Values, ODE_MAPK_Values, EC100_MAPK_ODE)

EC50_MAPKKK_HILL = EC50(E1Total_Values, HILL_MAPKKK_Values, EC100_MAPKKK_HILL)
EC50_MAPKK_HILL = EC50(E1Total_Values, HILL_MAPKK_Values, EC100_MAPKK_HILL)
EC50_MAPK_HILL = EC50(E1Total_Values, HILL_MAPK_Values, EC100_MAPK_HILL)

# Define multiples and initialize arrays to store normalized steady-state values
Multiples = np.linspace(0, 5, 100)

ODE_MAPKKK, ODE_MAPKK, ODE_MAPK = [], [], []
HILL_MAPKKK, HILL_MAPKK, HILL_MAPK = [], [], []

# Loop over multiples of EC50 and calculate the normalized steady states
for i in Multiples:
    E1_MAPKKK = i * EC50_MAPKKK_ODE
    E1_MAPKK = i * EC50_MAPKK_ODE
    E1_MAPK = i * EC50_MAPK_ODE

    E1_MAPKKK_HILL = i * EC50_MAPKKK_HILL
    E1_MAPKK_HILL = i * EC50_MAPKK_HILL
    E1_MAPK_HILL = i * EC50_MAPK_HILL

    SS_MAPKKK_ODE, SS_MAPKK_ODE, SS_MAPK_ODE = Kinase_SteadyState(E1_MAPKKK)
    ODE_MAPKKK.append(SS_MAPKKK_ODE / EC100_MAPKKK_ODE)

    SS_MAPKKK_HILL = Hill(E1_MAPKKK_HILL, 0.3, 1)
    HILL_MAPKKK.append(SS_MAPKKK_HILL / EC100_MAPKKK_HILL)
    
    SS_MAPKKK_ODE, SS_MAPKK_ODE, SS_MAPK_ODE = Kinase_SteadyState(E1_MAPKK)
    ODE_MAPKK.append(SS_MAPKK_ODE / EC100_MAPKK_ODE)

    SS_MAPKK_HILL = Hill(E1_MAPKK_HILL, 0.3, 1.7)
    HILL_MAPKK.append(SS_MAPKK_HILL / EC100_MAPKK_HILL)

    SS_MAPKKK_ODE, SS_MAPKK_ODE, SS_MAPK_ODE = Kinase_SteadyState(E1_MAPK)
    ODE_MAPK.append(SS_MAPK_ODE / EC100_MAPK_ODE)

    SS_MAPK_HILL = Hill(E1_MAPK_HILL, 0.3, 4.9)
    HILL_MAPK.append(SS_MAPK_HILL / EC100_MAPK_HILL)

# Plot the normalized steady-state kinase activity against multiples of EC50
plt.figure()
plt.plot(Multiples, ODE_MAPKKK, label = 'MAPKKK', color = 'darkblue')
plt.plot(Multiples, HILL_MAPKKK, '--', color = 'darkblue')

plt.plot(Multiples, ODE_MAPKK, label = 'MAPKK', color = 'cyan')
plt.plot(Multiples, HILL_MAPKK, '--', color = 'cyan')

plt.plot(Multiples, ODE_MAPK, label = 'MAPK', color = 'goldenrod')
plt.plot(Multiples, HILL_MAPK, '--', color = 'goldenrod')

plt.ylabel('Predicted Steady-State Kinase Activity')
plt.xlabel('Input Stimulus (E1 Total in Multiples of EC50)')
plt.legend()
plt.show()

# ------------------------- FIGURE 2B ------------------------------
E1_Concentrations = np.logspace(-6, -1, 100)
SS_MAPKKK_Values, SS_MAPKK_Values, SS_MAPK_Values = [], [], []
for E1 in E1_Concentrations:
    SS_MAPKKK, SS_MAPKK, SS_MAPK = Kinase_SteadyState(E1)
    SS_MAPKKK_Values.append(SS_MAPKKK / EC100_MAPKKK_ODE)
    SS_MAPKK_Values.append(SS_MAPKK / EC100_MAPKK_ODE)
    SS_MAPK_Values.append(SS_MAPK / EC100_MAPK_ODE)

plt.plot(E1_Concentrations, SS_MAPKKK_Values, label = 'MAPKKK', color = 'darkblue')
plt.plot(E1_Concentrations, SS_MAPKK_Values, label = 'MAPKK', color = 'cyan')
plt.plot(E1_Concentrations, SS_MAPK_Values, label = 'MAPK', color = 'goldenrod')

plt.axhline(y = 0.9, color = 'grey')
plt.axhline(y = 0.1, color = 'grey')

plt.xscale('log')
plt.ylabel('Predicted Steady-State Kinase Activity')
plt.xlabel('Input Stimulus (E1 Total)')
plt.legend()
plt.show()

# ------------------------- FIGURE 3A ------------------------------

plt.plot(SS_MAPKKK_Values, SS_MAPK_Values, label = 'ODE Model', color = 'darkblue')
plt.plot(HILL_MAPKKK_Values, HILL_MAPK_Values, label = 'Hill Model', color = 'cyan')
plt.xlabel('[malE-Mos] (uM)')
plt.ylabel('p42 MAP kinase (MAPK) activity')
plt.title('MAPK Activity vs MAPKKK Activity')

plt.legend() 
plt.show()
# ------------------------- FIGURE 3C ------------------------------
plt.figure()
plt.plot(SS_MAPKKK_Values, SS_MAPKK_Values, label = 'ODE Model', color = 'darkblue')
plt.plot(HILL_MAPKKK_Values, HILL_MAPKK_Values, label = 'Hill Model', color = 'cyan')
plt.xlabel('[malE-Mos] (uM)')
plt.ylabel('Mek-1 (MAPKK) activity')
plt.title('MAPKK Activity vs MAPKKK Activity')

plt.legend()
plt.show()