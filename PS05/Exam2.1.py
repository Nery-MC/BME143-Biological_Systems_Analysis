
# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Exam #2 - Problem #1 : Three-Step Phosphorylation Hypothesis

import numpy as np 
import matplotlib.pyplot as plt
import copy
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Create a new array that holds the initial values with additions for third phosphorylation state

y0 = np.zeros(28) # 24 old molcules/states + 6 new ones - 2 not needed (KK_PP Complex 1 and 2)
y0[0] = 0.1       # E1 
y0[1] = 0         # E1 Complex 
y0[2] = 0.003     # MAPKKK
y0[3] = 0         # MAPKKK*
y0[4] = 0         # MAPKKK* Complex 1 
y0[5] = 0         # MAPKKK* Complex 2
y0[6] = 0.003     # E2 
y0[7] = 0         # E2 Complex 
y0[8] = 1.2       # MAPKK
y0[9] = 0         # MAPKK_P
y0[10] = 0        # MAPKK_PP
y0[11] = 0.0003   # MAPKK_Pase 
y0[12] = 0        # MAPKK_Pase Complex 
y0[13] = 0.0003   # MAPKK_PPase 
y0[14] = 0        # MAPKK_PPase Complex 
y0[15] = 1.2      # MAPK
y0[16] = 0        # MAPK_P
y0[17] = 0        # MAPK_PP
y0[18] = 0.12     # MAPK_Pase 
y0[19] = 0        # MAPK_Pase Complex 
y0[20] = 0.12     # MAPK_PPase 
y0[21] = 0        # MAPK_PPase Complex 
# ========================== NEW ================================
y0[22] = 0        # MAPKKK* Complex 3 
y0[23] = 0        # MAPKK_PPP 
y0[24] = 0        # MAPKK_PPP Complex 1 
y0[25] = 0        # MAPKK_PPP Complex 2 
y0[26] = 0.0003   # MAPKK_PPPase 
y0[27] = 0        # MAPKK_PPPase Complex 
# ===============================================================
Time = [0, 100]

# Define the ODE function
def MAPKinase_ODE(t, y):

    dydt = np.zeros(28)
    # Rewriting the differential equations with corrected indices
    dydt[0] = -1000 * y[2] * y[0] + 150 * y[1] + 150 * y[1]
    dydt[1] = +1000 * y[2] * y[0] - 150 * y[1] - 150 * y[1]
    dydt[2] = -1000 * y[2] * y[0] + 150 * y[1] + 150 * y[7]
    dydt[3] = (-1000 * y[8] * y[3] + 150 * y[4] + 150 * y[4]
                - 1000 * y[9] * y[3] + 150 * y[5] + 150 * y[5]
                + 150 * y[1] - 1000 * y[3] * y[6] + 150 * y[7]
                + 150 * y[22] + 150 * y[22])
    dydt[4] = +1000 * y[8] * y[3] - 150 * y[4] - 150 * y[4]
    dydt[5] = +1000 * y[9] * y[3] - 150 * y[5] - 150 * y[5]
    dydt[6] = -1000 * y[3] * y[6] + 150 * y[7] + 150 * y[7]
    dydt[7] = +1000 * y[3] * y[6] - 150 * y[7] - 150 * y[7]
    dydt[8] = -1000 * y[8] * y[3] + 150 * y[4] + 150 * y[12]
    dydt[9] = (+150 * y[4] - 1000 * y[9] * y[3] + 150 * y[5]
                - 1000 * y[9] * y[11] + 150 * y[12] + 150 * y[14])
    dydt[10] = (-1000 * y[15] * y[10] - 1000 * y[16] * y[10]
                + 150 * y[5] - 1000 * y[10] * y[13] + 150 * y[14])
    dydt[11] = -1000 * y[9] * y[11] + 150 * y[12] + 150 * y[12]
    dydt[12] = +1000 * y[9] * y[11] - 150 * y[12] - 150 * y[12]
    dydt[13] = -1000 * y[10] * y[13] + 150 * y[14] + 150 * y[14]
    dydt[14] = +1000 * y[10] * y[13] - 150 * y[14] - 150 * y[14]
    dydt[15] = -1000 * y[15] * y[23] + 150 * y[19] + 150 * y[24]
    dydt[16] = (-1000 * y[16] * y[23] + 150 * y[24] + 150 * y[25]  
                + 150 * y[21] - 1000 * y[16] * y[18] + 150 * y[19])
    dydt[17] = - 1000 * y[17] * y[20] + 150 * y[25] + 150 * y[21]
    dydt[18] = -1000 * y[16] * y[18] + 150 * y[19] + 150 * y[19]
    dydt[19] = +1000 * y[16] * y[18] - 150 * y[19] - 150 * y[19]
    dydt[20] = -1000 * y[17] * y[20] + 150 * y[21] + 150 * y[21]
    dydt[21] = +1000 * y[17] * y[20] - 150 * y[21] - 150 * y[21]
    # ========================== ADDITIONAL ================================
    dydt[22] = +1000 * y[10] * y[3] - 150 * y[22] - 150 * y[22]
    dydt[23] = (-1000 * y[15] * y[23] + 150 * y[24] + 150 * y[24] 
                - 1000 * y[16] * y[23] + 150 * y[25] + 150 * y[22]
                - 1000 * y[26] * y[23] + 150 * y[27])
    dydt[24] = +1000 * y[23] * y[15] - 150 * y[24] - 150 * y[24]
    dydt[25] = +1000 * y[23] * y[16] - 150 * y[25] - 150 * y[25]
    dydt[26] = -1000 * y[23] * y[26] + 150 * y[27] 
    dydt[27] = +1000 * y[23] * y[26] - 150 * y[27] - 150 * y[27]
    # ======================================================================
    return dydt

# ------------------------------ RUN CODE -------------------------------------------
# Define the initial parameters
E1_Concentrations = np.logspace(-6, 1, 100)

# Function to extract the steady state of each kinase from solving the ODE 
def Kinase_SteadyState(E1_concentration):
    y0[0] = E1_concentration
    Solution = solve_ivp(MAPKinase_ODE, [0, 100], y0, method = 'BDF', t_eval = np.linspace(Time[0], Time[1], 1000))
    SS_MAPKKK, SS_MAPKK, SS_MAPK = Solution.y[3, -1], Solution.y[23, -1], Solution.y[17, -1]
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