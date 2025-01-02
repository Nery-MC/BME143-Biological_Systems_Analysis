import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Molecule names
molnames = [
    '1 /kinetics/MAPK/INPUT(E1)',
    '2 /kinetics/MAPK/INPUT(E1)/INPUT(E1)/INPUT(E1)_cplx',
    '3 /kinetics/MAPK/MAPKKK',
    '4 /kinetics/MAPK/MAPKKK*',
    '5 /kinetics/MAPK/MAPKKK*/MAPKKK*/MAPKKK*_cplx',
    '6 /kinetics/MAPK/MAPKKK*/MAPKKK*[1]/MAPKKK*[1]_cplx',
    '7 /kinetics/MAPK/E2',
    '8 /kinetics/MAPK/E2/E2/E2_cplx',
    '9 /kinetics/MAPK/MAPKK',
    '10 /kinetics/MAPK/MAPKK-P',
    '11 /kinetics/MAPK/MAPKK-PP',
    '12 /kinetics/MAPK/MAPKK-PP/MAPKK-PP/MAPKK-PP_cplx',
    '13 /kinetics/MAPK/MAPKK-PP/MAPKK-PP[1]/MAPKK-PP[1]_cplx',
    '14 /kinetics/MAPK/MAPKKPase',
    '15 /kinetics/MAPK/MAPKKPase/MAPKKPase/MAPKKPase_cplx',
    '16 /kinetics/MAPK/MAPKKPase[1]',
    '17 /kinetics/MAPK/MAPKKPase[1]/MAPKKPase/MAPKKPase_cplx',
    '18 /kinetics/MAPK/MAPK',
    '19 /kinetics/MAPK/MAPK-P',
    '20 /kinetics/MAPK/MAPK-PP',
    '21 /kinetics/MAPK/MAPKPase',
    '22 /kinetics/MAPK/MAPKPase/MAPKPase/MAPKPase_cplx',
    '23 /kinetics/MAPK/MAPKPase[1]',
    '24 /kinetics/MAPK/MAPKPase[1]/MAPKPase/MAPKPase_cplx',
]


# Initial concentrations of 24 molecules
y0 = np.zeros(24)
y0[0] = 0.1       # '1 /kinetics/MAPK/INPUT(E1)'
y0[1] = 0         # '2 /kinetics/MAPK/INPUT(E1)/INPUT(E1)/INPUT(E1)_cplx'
y0[2] = 0.003     # '3 /kinetics/MAPK/MAPKKK'
y0[3] = 0         # '4 /kinetics/MAPK/MAPKKK*'
y0[4] = 0         # '5 /kinetics/MAPK/MAPKKK*/MAPKKK*/MAPKKK*_cplx'
y0[5] = 0         # '6 /kinetics/MAPK/MAPKKK*/MAPKKK*[1]/MAPKKK*[1]_cplx'
y0[6] = 0.0003    # '7 /kinetics/MAPK/E2'
y0[7] = 0         # '8 /kinetics/MAPK/E2/E2/E2_cplx'
y0[8] = 1.2       # '9 /kinetics/MAPK/MAPKK'
y0[9] = 0         # '10 /kinetics/MAPK/MAPKK-P'
y0[10] = 0        # '11 /kinetics/MAPK/MAPKK-PP'
y0[11] = 0        # '12 /kinetics/MAPK/MAPKK-PP/MAPKK-PP/MAPKK-PP_cplx'
y0[12] = 0        # '13 /kinetics/MAPK/MAPKK-PP/MAPKK-PP[1]/MAPKK-PP[1]_cplx'
y0[13] = 0.0003   # '14 /kinetics/MAPK/MAPKKPase'
y0[14] = 0        # '15 /kinetics/MAPK/MAPKKPase/MAPKKPase/MAPKKPase_cplx'
y0[15] = 0.0003   # '16 /kinetics/MAPK/MAPKKPase[1]'
y0[16] = 0        # '17 /kinetics/MAPK/MAPKKPase[1]/MAPKKPase/MAPKKPase_cplx'
y0[17] = 1.2      # '18 /kinetics/MAPK/MAPK'
y0[18] = 0        # '19 /kinetics/MAPK/MAPK-P'
y0[19] = 0        # '20 /kinetics/MAPK/MAPK-PP'
y0[20] = 0.12     # '21 /kinetics/MAPK/MAPKPase'
y0[21] = 0        # '22 /kinetics/MAPK/MAPKPase/MAPKPase/MAPKPase_cplx'
y0[22] = 0.12     # '23 /kinetics/MAPK/MAPKPase[1]'
y0[23] = 0        # '24 /kinetics/MAPK/MAPKPase[1]/MAPKPase/MAPKPase_cplx'

tspan = [0, 100]

# Define the ODE function
def f(t, y):
    dydt = np.zeros(24)
    # Rewriting the differential equations with corrected indices
    dydt[0] = -1000 * y[2] * y[0] + 150 * y[1] + 150 * y[1]
    dydt[1] = +1000 * y[2] * y[0] - 150 * y[1] - 150 * y[1]
    dydt[2] = -1000 * y[2] * y[0] + 150 * y[1] + 150 * y[7]
    dydt[3] = (-1000 * y[8] * y[3] + 150 * y[4] + 150 * y[4]
                - 1000 * y[9] * y[3] + 150 * y[5] + 150 * y[5]
                + 150 * y[1] - 1000 * y[3] * y[6] + 150 * y[7])
    dydt[4] = +1000 * y[8] * y[3] - 150 * y[4] - 150 * y[4]
    dydt[5] = +1000 * y[9] * y[3] - 150 * y[5] - 150 * y[5]
    dydt[6] = -1000 * y[3] * y[6] + 150 * y[7] + 150 * y[7]
    dydt[7] = +1000 * y[3] * y[6] - 150 * y[7] - 150 * y[7]
    dydt[8] = -1000 * y[8] * y[3] + 150 * y[4] + 150 * y[14]
    dydt[9] = (+150 * y[4] - 1000 * y[9] * y[3] + 150 * y[5]
                - 1000 * y[9] * y[13] + 150 * y[14] + 150 * y[16])
    dydt[10] = (-1000 * y[17] * y[10] + 150 * y[11] + 150 * y[11]
                - 1000 * y[18] * y[10] + 150 * y[12] + 150 * y[12]
                + 150 * y[5] - 1000 * y[10] * y[15] + 150 * y[16])
    dydt[11] = +1000 * y[17] * y[10] - 150 * y[11] - 150 * y[11]
    dydt[12] = +1000 * y[18] * y[10] - 150 * y[12] - 150 * y[12]
    dydt[13] = -1000 * y[9] * y[13] + 150 * y[14] + 150 * y[14]
    dydt[14] = +1000 * y[9] * y[13] - 150 * y[14] - 150 * y[14]
    dydt[15] = -1000 * y[10] * y[15] + 150 * y[16] + 150 * y[16]
    dydt[16] = +1000 * y[10] * y[15] - 150 * y[16] - 150 * y[16]
    dydt[17] = -1000 * y[17] * y[10] + 150 * y[11] + 150 * y[21]
    dydt[18] = (+150 * y[11] - 1000 * y[18] * y[10] + 150 * y[12]
                + 150 * y[23] - 1000 * y[18] * y[20] + 150 * y[21])
    dydt[19] = +150 * y[12] - 1000 * y[19] * y[22] + 150 * y[23]
    dydt[20] = -1000 * y[18] * y[20] + 150 * y[21] + 150 * y[21]
    dydt[21] = +1000 * y[18] * y[20] - 150 * y[21] - 150 * y[21]
    dydt[22] = -1000 * y[19] * y[22] + 150 * y[23] + 150 * y[23]
    dydt[23] = +1000 * y[19] * y[22] - 150 * y[23] - 150 * y[23]
    return dydt

# Solve the ODEs
sol = solve_ivp(
    f, tspan, y0, method='BDF', t_eval=np.linspace(tspan[0], tspan[1], 1000)
)

t = sol.t
y = sol.y


