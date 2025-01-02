
# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Problem Set #1 - Problem #1 : ODE System Dynamics 

from scipy.integrate import solve_ivp
import numpy as np 
import matplotlib.pyplot as plt

# Get user inputs for variables 
t = float(input("t = "))
A = float(input("A = "))
x0 = float(input("x = "))
y0 = float(input("y = "))

# Combine system of ODE equations 
def System(t, U, A): 
    x, y, = U 
    dx_dt = y # Equation 1 
    dy_dt = x - (x**3) - (0.25 * y) + (A * np.sin(t) * t) # Equation 2 
    return (dx_dt, dy_dt)

# Time span and initial conditions 
time = (0, t)
initial = (x0, y0)

# Solution to the ODE system 
solution = solve_ivp(System, time, initial, args = (A, ))

# Plot the values 
fig, ax = plt.subplots(2) 
fig.suptitle('ODE System')
ax[0].plot(solution.t, solution.y[0], 'tab:blue', label = "X(t)")
ax[1].plot(solution.t, solution.y[1], 'tab:orange', label = "Y(t)")

for i in ax.flat: 
    i.set(xlabel = "t [Time]" , ylabel = "[Solution]")

ax[0].legend() 
ax[0].grid() 

ax[1].legend()
ax[1].grid()

plt.tight_layout()
plt.show()