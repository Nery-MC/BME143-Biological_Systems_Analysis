# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Date : December 20, 2024
# Microbiome-Pathogen Interactions Drive Epidemiological Dynamics of ABR

# Final Project Model Implementation : Figure 1.py
# --------------------------------------------------------------------------------------------
from Values import Epsilon_High, Epsilon_Medium, Epsilon_Low, Phi_High, Phi_Medium, Phi_Low, Eta_High, Eta_Medium, Eta_Low
from Values import PerfectR_Epsilon_High, PerfectR_Epsilon_Medium, PerfectR_Epsilon_Low, PerfectR_Eta_High, PerfectR_Eta_Medium, PerfectR_Eta_Low, PerfectR_Phi_High, PerfectR_Phi_Medium, PerfectR_Phi_Low

import matplotlib.pyplot as plt
import numpy as np
from Function import Univar_Sweep
from Model_ODEs import Susceptible_Colonized_ODE, Strain_Competition_ODE, Microbiome_Competition_ODE, TwoStrainBiome_Competition_ODE, TwoStrainHGT_Competition_ODE
from Values import State_1, State_2, State_3, State_4, State_5, p, PerfectR

# ----------------------------------------- FIGURE1 -------------------------------------------
# Solve the ODE system as a function of 'a' (Antibiotic Exposure Prevelance)
a = np.linspace(0, 1, 100)

# MODEL 1

## rR = 0.8
FIG1_MODEL1A = Univar_Sweep(9, a, 'Model_1', Susceptible_Colonized_ODE, State_1, p)
## rR = 1
FIG1_MODEL1B = Univar_Sweep(9, a, 'Model_1', Susceptible_Colonized_ODE, State_1, PerfectR)

# MODEL 2

## rR = 0.8
FIG1_MODEL2A = Univar_Sweep(9, a, 'Model_2', Strain_Competition_ODE, State_2, p)
## rR = 1
FIG1_MODEL2B = Univar_Sweep(9, a, 'Model_2', Strain_Competition_ODE, State_2, PerfectR)

# MODEL 3

## rR = 0.8
### EPSILON
FIG1_MODEL3A_EPSLOW = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Epsilon_Low)

FIG1_MODEL3A_EPSMED = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Epsilon_Medium)

FIG1_MODEL3A_EPSHIGH = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Epsilon_High)

### ETA 
FIG1_MODEL3A_ETALOW = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Eta_Low)

FIG1_MODEL3A_ETAMED = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Eta_Medium)

FIG1_MODEL3A_ETAHIGH = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Eta_High)

### PHI 
FIG1_MODEL3A_PHILOW = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Phi_Low)

FIG1_MODEL3A_PHIMED = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Phi_Medium)

FIG1_MODEL3A_PHIHIGH = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, Phi_High)

FIG1_MODEL3A_INTERACTIONS = (FIG1_MODEL3A_EPSLOW, FIG1_MODEL3A_EPSMED, FIG1_MODEL3A_EPSHIGH,
                             FIG1_MODEL3A_ETALOW, FIG1_MODEL3A_ETAMED, FIG1_MODEL3A_ETAHIGH,
                             FIG1_MODEL3A_PHILOW, FIG1_MODEL3A_PHIMED, FIG1_MODEL3A_PHIHIGH)

## rR = 1
### EPSILON
FIG1_MODEL3B_EPSLOW = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Epsilon_Low)

FIG1_MODEL3B_EPSMED = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Epsilon_Medium)

FIG1_MODEL3B_EPSHIGH = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Epsilon_High)

### ETA 
FIG1_MODEL3B_ETALOW = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Eta_Low)

FIG1_MODEL3B_ETAMED = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Eta_Medium)

FIG1_MODEL3B_ETAHIGH = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Eta_High)

### PHI 
FIG1_MODEL3B_PHILOW = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Phi_Low)

FIG1_MODEL3B_PHIMED = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Phi_Medium)

FIG1_MODEL3B_PHIHIGH = Univar_Sweep(9, a, 'Model_3', Microbiome_Competition_ODE, State_3, PerfectR_Phi_High)

FIG1_MODEL3B_INTERACTIONS = (FIG1_MODEL3B_EPSLOW, FIG1_MODEL3B_EPSMED, FIG1_MODEL3B_EPSHIGH,
                             FIG1_MODEL3B_ETALOW, FIG1_MODEL3B_ETAMED, FIG1_MODEL3B_ETAHIGH,
                             FIG1_MODEL3B_PHILOW, FIG1_MODEL3B_PHIMED, FIG1_MODEL3B_PHIHIGH)

# PLOTTING FUNCTIONS

def Find_Max(Data, X_Parameter, Y_Parameter, ax, color): 
    MAX_Y_Index = Data[Y_Parameter].idxmax()
    MAX_X_Value = Data.loc[MAX_Y_Index, X_Parameter]
    MAX_Y_Value = Data.loc[MAX_Y_Index, Y_Parameter]
    ax.plot([MAX_X_Value, MAX_X_Value], [0, MAX_Y_Value], color = color, linestyle='--')

a_Label = 'Antibiotic Exposure Prevalence (a)'
Prevalence_Label = 'Colonization Prevalence (C^R)'
Incidence_Label = 'Daily Colonization Incidence (% of Patients)'
R_Rate_Label = 'Resistance Rate (C^R / C^S + C^R)'

fig, ax = plt.subplots(1, 3) 
ax[0].plot(FIG1_MODEL1A['Parameter_Value'], FIG1_MODEL1A['Prevalence_CR'], linestyle = '-', color = 'black')
Find_Max(FIG1_MODEL1A, 'Parameter_Value', 'Prevalence_CR', ax[0], 'black')
ax[0].set_title('Model 1')

ax[1].plot(FIG1_MODEL2A['Parameter_Value'], FIG1_MODEL2A['Prevalence_CR'], linestyle = '-', color = 'red')
ax[1].plot(FIG1_MODEL2A['Parameter_Value'], FIG1_MODEL2A['Prevalence_CS'], linestyle = '-', color = 'grey')
Find_Max(FIG1_MODEL2A, 'Parameter_Value', 'Prevalence_CR', ax[1], 'red')
ax[1].set_title('Model 2')

ax[2].plot(FIG1_MODEL3A_EPSMED['Parameter_Value'], FIG1_MODEL3A_EPSMED['Prevalence_CR'], linestyle = '-', color = 'green')
ax[2].plot(FIG1_MODEL3A_ETAMED['Parameter_Value'], FIG1_MODEL3A_ETAMED['Prevalence_CR'], linestyle = '-', color = 'purple')
ax[2].plot(FIG1_MODEL3A_PHIMED['Parameter_Value'], FIG1_MODEL3A_PHIMED['Prevalence_CR'], linestyle = '-', color = 'pink')
Find_Max(FIG1_MODEL3A_EPSMED, 'Parameter_Value', 'Prevalence_CR', ax[2], 'green')
Find_Max(FIG1_MODEL3A_ETAMED, 'Parameter_Value', 'Prevalence_CR', ax[2], 'purple')
Find_Max(FIG1_MODEL3A_PHIMED, 'Parameter_Value', 'Prevalence_CR', ax[2], 'pink')
ax[2].set_title('Model 3')

for axis in ax:
    axis.set(xlabel = a_Label, ylabel = Prevalence_Label)

    axis.set_ylim(0, 0.5)
    axis.label_outer()

plt.show()