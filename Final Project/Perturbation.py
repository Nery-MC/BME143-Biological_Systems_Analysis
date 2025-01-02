# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Date : December 20, 2024
# Microbiome-Pathogen Interactions Drive Epidemiological Dynamics of ABR

# Final Project Model Implementation : Perturbation.py
# --------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt 
from scipy.integrate import solve_ivp

# Define the ODE system 
def Microbiome_ODE(t, U, Parameters):
    Se, Sd, CRe, CRd = U
    Lmd_Re, Lmd_Rd, Gam_R, Sig_M, D, Mu, a_R, q_M = Parameters

    dSe_dt = - Lmd_Re * Se + D * Sd - Sig_M * Se - Mu * Se
    dSd_dt = Sig_M * Se - Lmd_Rd * Sd + D * CRd - Mu * Sd
    dCRe_dt = Lmd_Re * Se - (Gam_R + Sig_M) * CRe + D * CRd - Mu * CRe
    dCRd_dt = Lmd_Rd * Sd - (Gam_R + D) * CRd + Sig_M * CRe - Mu * CRd

    return [dSe_dt, dSd_dt, dCRe_dt, dCRd_dt]

# Initial conditions
Se0 = 0.9  # Proportion of susceptible individuals with stable microbiome 
Sd0 = 0.1  # Proportion of susceptible individuals with distrupted microbiome 
CRe0 = 0   # Proportion of colonized individuals with stable microbiome 
CRd0 = 0   # Proportion of colonized individuals with distrupted microbiome

Initial = [Se0, Sd0, CRe0, CRd0]

# Parameters 
Parameters = [
    0.02,  # Lmd_Re   Colonization rate for stable microbiome 
    0.05,  # Lmd_Rd   Colonization rate for distrupted microbiome
    0.01,  # Gam_R    Clearance rate 
    0.1,   # Sig_M    Dysbiosis rate 
    0.02,  # D        Microbiome recovery rate 
    0.01,  # Mu       Admission/Discharge rate 
    0.005, # a_R      Endogenous acquisition rate 
    0.05   # q_M      Antibiotic-induced dysbiosis rate 
]

# Time span for simulation 
Time = (0, 200)   # From day 0 to day 200
Time_Span = np.linspace(0, 200, 1000)

# ---------------------------- Initial Solution : No Perturbations --------------------------------------------
# Initial solve of ODE 
Solution = solve_ivp(Microbiome_ODE, Time, Initial, args = (Parameters, ), t_eval = Time_Span)
# Extract solution 
Se, Sd, CRe, CRd = Solution.y 
# Plot the results
plt.figure()
plt.plot(Solution.t, Se, label = 'Susceptible (Stable Microbiome)')
plt.plot(Solution.t, Sd, label = 'Susceptible (Disrupted Microbiome)')
plt.plot(Solution.t, CRe, label = 'Colonized (Stable Microbiome)')
plt.plot(Solution.t, CRd, label = 'Colonized (Disrupted Microbiome)')
plt.xlabel('Time (days)')
plt.ylabel('Proportion of Population')
plt.title('Dynamics of Colonization and Microbiome Disruption')
plt.legend()
plt.grid()
plt.show()
# -------------------------------------------------------------------------------------------------------------
# Function to perform parameter sweeps 
def Parameter_Sweep(i, Values, Initial_Conditions, Parameters, Time, Time_Span):
    Results = []
    for value in Values:
        # Update the parameter of interest 
        Parameters[i] = value 

        # Solve the ODE 
        Solution = solve_ivp(Microbiome_ODE, Time, Initial_Conditions, args = (Parameters, ), t_eval = Time_Span)

        Results.append((value, Solution.t, Solution.y))

    return Results

# ---------------------------- Sweep : Antibiotic-Induced (AI) Dysbiosis Rate (Sig_M) --------------------------
Parameters = [0.02, 0.05, 0.01, 0.1, 0.02, 0.01, 0.005, 0.05]
Sig_M_Values = np.linspace(0.01, 0.2, 5) # Test values for Sigma M
AI_Dysbiosis_Results = Parameter_Sweep(3, Sig_M_Values, Initial, Parameters, Time, Time_Span)

# Plot the results 
plt.figure()
for Sig_M, t, y in AI_Dysbiosis_Results:
    Se, Sd, CRe, CRd = y
    plt.plot(Time_Span, CRe + CRd, label = f'Sig_M = {Sig_M:.2f}')

plt.xlabel('Time (Days)')
plt.ylabel('Proportion Colonized')
plt.title('Effect of Antibiotic-Induced Dysbiosis on Colonization')
plt.legend()
plt.grid()
plt.show()

# ---------------------------- Sweep : Microbiome Recovery Rate (D) --------------------------------------------
Parameters = [0.02, 0.05, 0.01, 0.1, 0.02, 0.01, 0.005, 0.05]
D_Values = np.linspace(0.01, 0.1, 5)  # Test values for recovery rate
Biome_Recovery_Results = Parameter_Sweep(4, D_Values, Initial, Parameters, Time, Time_Span)

# Plot the results for recovery dynamics
plt.figure()
for D, t, y in Biome_Recovery_Results:
    Se, Sd, CRe, CRd = y
    plt.plot(t, CRe + CRd, label = f'D = {D:.2f}')

plt.xlabel('Time (days)')
plt.ylabel('Proportion Colonized')
plt.title('Effect of Microbiome Recovery Rate on Colonization')
plt.legend()
plt.grid()
plt.show()
# ---------------------------- Sweep : Colonization Rate - Stable Microbiome (Lambda_Re) --------------------------
Parameters = [0.02, 0.05, 0.01, 0.1, 0.02, 0.01, 0.005, 0.05]
Lmd_Re_Values = np.linspace(0.01, 0.1, 5)
Colonization_Stable_Results = Parameter_Sweep(0, Lmd_Re_Values, Initial, Parameters, Time, Time_Span)

# Plot the results for colonization dynamics 
plt.figure()
for Lmd_Re, t, y in Colonization_Stable_Results:
    Se, Sd, CRe, CRd = y
    plt.plot(t, CRe + CRd, label = f'Lambda_Re = {Lmd_Re:.2f}')

plt.xlabel('Time (days)')
plt.ylabel('Proportion Colonized')
plt.title('Effect of Colonization Rate (Stable Microbiome) on Colonization')
plt.legend()
plt.grid()
plt.show()

# ---------------------------- Sweep : Colonization Rate - Disturbed Microbiome (Lambda_Rd) -----------------------
Parameters = [0.02, 0.05, 0.01, 0.1, 0.02, 0.01, 0.005, 0.05]
Lambda_Rd_Values = np.linspace(0.01, 0.2, 5) 
Colonization_Disturb_Results = Parameter_Sweep(1, Lambda_Rd_Values, Initial, Parameters, Time, Time_Span)

# Plot the results for colonization dynamics 
plt.figure()
for Lmd_Rd, t, y in Colonization_Disturb_Results:
    Se, Sd, CRe, CRd = y
    plt.plot(t, CRe + CRd, label = f'Lambda_Rd = {Lmd_Rd:.2f}')

plt.xlabel('Time (days)')
plt.ylabel('Proportion Colonized')
plt.title('Effect of Colonization Rate (Disturbed Microbiome) on Colonization')
plt.legend()
plt.grid()
plt.show()

# ---------------------------- Sweep : Clearance Rate (Gamma_R) -----------------------------------------------------
Parameters = [0.02, 0.05, 0.01, 0.1, 0.02, 0.01, 0.005, 0.05]
Gam_R_Values = np.linspace(0.005, 0.05, 5) 
Clearance_Results = Parameter_Sweep(2, Gam_R_Values, Initial, Parameters, Time, Time_Span)

# Plot the results for clearance rate dynamics
plt.figure()
for Gam_R, t, y in Clearance_Results:
    Se, Sd, CRe, CRd = y
    plt.plot(t, CRe + CRd, label = f'Gamma_R = {Gam_R:.2f}')

plt.xlabel('Time (days)')
plt.ylabel('Proportion Colonized')
plt.title('Effect of Clearance Rate on Colonization')
plt.legend()
plt.grid()
plt.show()
