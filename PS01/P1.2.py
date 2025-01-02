
# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Problem Set #1 - Problem #2 : Hill Model Pharmacodynamics

import numpy as np 
import matplotlib.pyplot as plt 

# Initiate constant values  
Vmax = 12 # maximum rate of reaction 
Km = 10 # Michaelis constant 
n = 4 
S = np.linspace(0, 100, 500) # varying substrate concentrations from 0 to 100

# Create an array that holds the varying values for varying values
n_array =  np.array([0.2, 0.5, 1, 3, 4, 6, 7])
Vmax_array = np.array([0, 5, 10, 15, 20, 25, 30])
Km_array = np.array([0, 5, 10, 15, 20, 25, 30])

# Function for the Hill Equation 
def Hill(Vmax, Km, S, n): 
    return (Vmax * S**n) / (Km**n + S**n)

# PART [A] 
# Calculate V(s) for varying 'n' 
for i in n_array: 
   V_s = Hill(Vmax, Km, S, i)

   plt.plot(S, V_s, label = f'n = {i}') 

plt.xlabel('S [Substrate Concentration]')
plt.ylabel('V(S) [Reaction Rate]')
plt.title('Hill Model Function with Constant Vmax and Km')
plt.grid(True)
plt.legend()
plt.show()

# PART [B]
# Calculate V(s) for varying 'V_max' 
for i in Vmax_array: 
    V_s = Hill(i, Km, S, n)

    plt.plot(S, V_s, label = f'Vmax = {i}')

plt.xlabel('S [Substrate Concentration]')
plt.ylabel('V(S) [Reaction Rate]')
plt.title('Hill Model Function with Constant n and Km')
plt.grid(True)
plt.legend()
plt.show()

# Calculate V(s) for varying 'Km' 
for i in Km_array: 
    V_s = Hill(Vmax, i, S, n)

    plt.plot(S, V_s, label = f'Km = {i}')

plt.xlabel('S [Substrate Concentration]')
plt.ylabel('V(S) [Reaction Rate]')
plt.title('Hill Model Function with Constant n and Vmax')
plt.grid(True)
plt.legend()
plt.show()

# Calculate V(s) for varying both 'Km' and 'V_max' 
for Vmax in Vmax_array: 
    for Km in Km_array: 
        V_s = Hill(Vmax, Km, S, n)

        plt.plot(S, V_s,)

plt.xlabel('S [Substrate Concentration]')
plt.ylabel('V(S) [Reaction Rate]')
plt.title('Hill Model Function with Constant n')
plt.grid(True)
plt.legend()
plt.show()