# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Date : December 20, 2024
# Microbiome-Pathogen Interactions Drive Epidemiological Dynamics of ABR

# # Final Project Model Implementation : Values.py
# --------------------------------------------------------------------------------------------
import numpy as np

State_1 = [0.9, 0.1, 0, 0, 0, 0, 0 ]
State_2 = [0.8, 0.1, 0.1, 0, 0, 0, 0, 0]
State_3 = [0.7, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0]
State_4 = [0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0]
State_5 = [0.3, 0.03, 0.2, 0.02, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0]

p = np.zeros(21)

p[0] = 0.2     # beta 
p[1] = 0.01    # alpha
p[2] = 0.03    # gamma
p[3] = 1       # c

# Patient Demography
p[4] = 0.1     # mu
p[5] = 0       # fd
p[6] = 0.1     # fC
p[7] = 0.5     # fR
p[8] = 0       # fw

# Antibiotics 
p[9] = 0.2     # a
p[10] = 0      # rS
p[11] = 0.8    # rR
p[12] = 0.2    # theta_C
p[13] = 1      # theta_M

# Microbiome Ecology 
p[14] = 0.5    # epsilon
p[15] = 0.5    # eta
p[16] = 5      # phi
p[17] = 0.01   # chi_e
p[18] = 0.1    # chi_d 
p[19] = 1 / 7  # delta
p[20] = 0.01   # omega
# ------------------------------------- FIGURE 1 VALUES -----------------------------------
def FIG1_Perturbation(Values, Array): 
    Low, Medium, High = Values
    
    Low_Array = Array.copy()
    for i, value in enumerate(Low):
        Low_Array[i + 14] = value

    Medium_Array = Array.copy()
    for i, value in enumerate(Medium):
        Medium_Array[i + 14] = value
    
    High_Array = Array.copy()
    for i, value in enumerate(High):
        High_Array[i + 14] = value

    return [Low_Array, Medium_Array, High_Array]

PerfectR = p.copy() 
PerfectR[11] = 1 # rR

# Varying Epsilon (Colonization Resistance)
Varying_Epsilon = ((0.2, 0, 1), (0.5, 0, 1), (0.8, 0, 1)) # Each element = LOW, MED, HIGH
Epsilon_Low, Epsilon_Medium, Epsilon_High = FIG1_Perturbation(Varying_Epsilon, p) # Default
PerfectR_Epsilon_Low, PerfectR_Epsilon_Medium, PerfectR_Epsilon_High = FIG1_Perturbation(Varying_Epsilon, PerfectR) # Perfect rR 

# Varying Eta (Resource Competition)
Varying_Eta = ((0, 0.2, 1), (0, 0.5, 1), (0, 0.8, 1))
Eta_Low, Eta_Medium, Eta_High = FIG1_Perturbation(Varying_Eta, p) # Default
PerfectR_Eta_Low, PerfectR_Eta_Medium, PerfectR_Eta_High = FIG1_Perturbation(Varying_Eta, PerfectR) # PerfectR

# Varyinf Phi (Ecological Release)
Varying_Phi = ((0, 0, 2), (0, 0, 5), (0, 0, 8))
Phi_Low, Phi_Medium, Phi_High = FIG1_Perturbation(Varying_Phi, p) # Default
PerfectR_Phi_Low, PerfectR_Phi_Medium, PerfectR_Phi_High = FIG1_Perturbation(Varying_Phi, PerfectR) # PerfectR

# ------------------------------------- FIGURE 3 VALUES -----------------------------------
import numpy as np

def FIG3_Perturbations(Resistance, Values, Array): 
    Zero, Low, High = Values[:3]  # Unpack Values for nINT
    Zero1, Low1, High1 = Values[3:]  # Unpack Values for wINT

    # Deep copies of the array
    results = {
        "nINT_nHGT": Array.copy(),
        "nINT_lowHGT": Array.copy(),
        "nINT_highHGT": Array.copy(),
        "wINT_nHGT": Array.copy(),
        "wINT_lowHGT": Array.copy(),
        "wINT_highHGT": Array.copy(),
    }

    # Helper function to update arrays
    def update_array(target, values, resistance):
        for i, value in enumerate(values):
            target[i + 14] = value  # Adjust the index as needed
        target[11] = resistance

    # Update arrays for each category
    update_array(results["nINT_nHGT"], Zero, Resistance)
    update_array(results["nINT_lowHGT"], Low, Resistance)
    update_array(results["nINT_highHGT"], High, Resistance)
    update_array(results["wINT_nHGT"], Zero1, Resistance)
    update_array(results["wINT_lowHGT"], Low1, Resistance)
    update_array(results["wINT_highHGT"], High1, Resistance)

    # Return arrays as a list
    return list(results.values())

Default = [
    [0, 0, 1, 0, 0], [0, 0, 1, 0.01, 0.1], [0, 0, 1, 0.1, 1], 
    [0.5, 0.5, 5, 0, 0], [0.5, 0.5, 5, 0.01, 0.1], [0.5, 0.5, 5, 0.1, 1]
]

# High Resistance
HR = 0.8
High_Resistance = FIG3_Perturbations(HR, Default, p)
HR_nINT_nHGT, HR_nINT_lowHGT, HR_nINT_highHGT, HR_wINT_nHGT, HR_wINT_lowHGT, HR_wINT_highHGT = High_Resistance

# Medium Resistance
MR = 0.5
Medium_Resistance = FIG3_Perturbations(MR, Default, p)
MR_nINT_nHGT, MR_nINT_lowHGT, MR_nINT_highHGT, MR_wINT_nHGT, MR_wINT_lowHGT, MR_wINT_highHGT = Medium_Resistance

# Low Resistance
LR = 0.2
Low_Resistance = FIG3_Perturbations(LR, Default, p)
LR_nINT_nHGT, LR_nINT_lowHGT, LR_nINT_highHGT, LR_wINT_nHGT, LR_wINT_lowHGT, LR_wINT_highHGT = Low_Resistance