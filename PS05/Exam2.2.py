
# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Exam #2 - Problem #2 : Ligand Receptor Complex 

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

Nav = 6.022 * 10e23 # [molecules/mole] Avogadro's number 

# ====================== PART [A] =============================
def RLBinding(t, U, ka, kd): 
    R, C, L = U 
    dRdt = (kd * C) - (ka * R * L)
    dCdt = (ka * R * L) - (kd * C) 
    dLdt = (kd * C) - (ka * R * L)
    return [dRdt, dCdt, dLdt]

# ====================== PART [B] =============================
# Rt = R + C # total number of receptors 
# Lt = L0 + C # total ligand concentration 

# ====================== PART [C] =============================
# Since conversion equations represent constant values 
def C_ODE(t, C, Rt, L0, ka, kd): 
    R = Rt - C
    L = L0 - C
    dCdt = (ka * R * L) - (kd * C)
    return dCdt

# ====================== PART [D] =============================
Conditions = 'TNF-a, TNFR, A549 Human Epithelial'
Rt = 6.6 * 10e3   # Receptors per cell 
ka = 9.6 * 10e8   # [1/min/M]
kd = 0.14         # [1/min]
Kd = 1.5 * 10e-10 # [M]
n = 10e5          # Total number of cells per well
Volume = 0.0002   # [L] 

Ligand_Concentration = np.arange(10e-12, 10e-8 + 10e-12, 10e-12)

# Calculate total receptors in the well
R = Rt * n 
R = (R) / (Nav * Volume) # Convert to [M]

# At equillibrium the foward reaction is equal to the reverse reaction
Ceq_Concentration = np.zeros(len(Ligand_Concentration))
for i, L in enumerate(Ligand_Concentration): 
    Ceq = (R * L) / Kd
    Ceq_Concentration[i] = Ceq

# Plot the results 
plt.plot(Ligand_Concentration, Ceq_Concentration, label = 'TNF-a/TNFR Complex')
plt.xlabel(' Initial Ligand Concentration (L0) [M]')
plt.ylabel('Equillibrium Complex Concentration (Ceq) [M]')
plt.title(Conditions)
plt.grid(True)
plt.legend()
plt.show()

# ====================== PART [E] =============================
Time = (0, 10)    # time from 0 to 10 minutes
time_span = np.linspace(0, 10, 500)
C0 = [0]  # Complex initial concentration


L01 = 3.5 * 10e-9 # [M]
Solution1 = solve_ivp(C_ODE, Time, C0, args = (R, L01, ka, kd), t_eval = time_span, method = 'BDF')

L02 = 2.5 * 10e-11 # [M]
Solution2 = solve_ivp(C_ODE, Time, C0, args = (R, L02, ka, kd), t_eval = time_span, method = 'BDF')
plt.figure() 
plt.subplot(1, 2, 1)
plt.plot(Solution1.t, Solution1.y[0], color = 'blue')
plt.xlabel('Time [min]')
plt.ylabel('Complex Concentration (C) [M]')
plt.title('Change in Complex Concetration [L0 = 3.5e-9 M]')
plt.grid(True)


plt.subplot(1, 2, 2)

plt.plot(Solution2.t, Solution2.y[0], color = 'red')
plt.xlabel('Time [min]')
plt.ylabel('Complex Concentration (C) [M]')
plt.title('Change in Complex Concetration [L0 = 2.5e-11 M]')
plt.grid(True)

plt.tight_layout()
plt.show()



