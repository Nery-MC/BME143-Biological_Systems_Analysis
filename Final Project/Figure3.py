# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Date : December 20, 2024
# Microbiome-Pathogen Interactions Drive Epidemiological Dynamics of ABR

# Final Project Model Implementation : Figure 3.py
# --------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from Function import Univar_Sweep
from Model_ODEs import TwoStrainHGT_Competition_ODE
from Values import State_1, State_2, State_3, State_4, State_5, p, PerfectR

a = np.linspace(0, 1, 100)
# ----------------------------------------- FIGURE 3 -------------------------------------------
from Values import High_Resistance, Medium_Resistance, Low_Resistance

HR_nINT_nHGT, HR_nINT_lowHGT, HR_nINT_highHGT, HR_wINT_nHGT, HR_wINT_lowHGT, HR_wINT_highHGT = High_Resistance
MR_nINT_nHGT, MR_nINT_lowHGT, MR_nINT_highHGT, MR_wINT_nHGT, MR_wINT_lowHGT, MR_wINT_highHGT = Medium_Resistance
LR_nINT_nHGT, LR_nINT_lowHGT, LR_nINT_highHGT, LR_wINT_nHGT, LR_wINT_lowHGT, LR_wINT_highHGT = Low_Resistance

### High Resistance 
FIG3_nINT_nHGT_HR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, HR_nINT_nHGT)
FIG3_nINT_lowHGT_HR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, HR_nINT_lowHGT)
FIG3_nINT_highHGT_HR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, HR_nINT_highHGT)
FIG3_wINT_nHGT_HR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, HR_wINT_nHGT)
FIG3_wINT_lowHGT_HR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, HR_wINT_lowHGT)
FIG3_wINT_highHGT_HR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, HR_wINT_highHGT)

# Each element = (Results, HGT Level, Interactions?)
FIG3A_INTERACTIONS = ((FIG3_nINT_nHGT_HR, 'null', 'none'), (FIG3_nINT_lowHGT_HR, 'low', 'none'), (FIG3_nINT_highHGT_HR, 'high', 'none'), 
                      (FIG3_wINT_nHGT_HR, 'null', 'yes'), (FIG3_wINT_lowHGT_HR, 'low', 'yes'), (FIG3_wINT_highHGT_HR, 'high', 'yes'))

### Medium Resistance 
FIG3_nINT_nHGT_MR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, MR_nINT_nHGT)
FIG3_nINT_lowHGT_MR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, MR_nINT_lowHGT)
FIG3_nINT_highHGT_MR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, MR_nINT_highHGT)
FIG3_wINT_nHGT_MR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, MR_wINT_nHGT)
FIG3_wINT_lowHGT_MR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, MR_wINT_lowHGT)
FIG3_wINT_highHGT_MR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, MR_wINT_highHGT)

FIG3B_INTERACTIONS = ((FIG3_nINT_nHGT_MR, 'null', 'none'), (FIG3_nINT_lowHGT_MR, 'low', 'none'), (FIG3_nINT_highHGT_MR, 'high', 'none'), 
                      (FIG3_wINT_nHGT_MR, 'null', 'yes'), (FIG3_wINT_lowHGT_MR, 'low', 'yes'), (FIG3_wINT_highHGT_MR, 'high', 'yes'))


### Low Resistance 
FIG3_nINT_nHGT_LR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, LR_nINT_nHGT)
FIG3_nINT_lowHGT_LR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, LR_nINT_lowHGT)
FIG3_nINT_highHGT_LR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, LR_nINT_highHGT)
FIG3_wINT_nHGT_LR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, LR_wINT_nHGT)
FIG3_wINT_lowHGT_LR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, LR_wINT_lowHGT)
FIG3_wINT_highHGT_LR = Univar_Sweep(9, a, 'Model_5', TwoStrainHGT_Competition_ODE, State_5, LR_wINT_highHGT)

FIG3B_INTERACTIONS = ((FIG3_nINT_nHGT_LR, 'null', 'none'), (FIG3_nINT_lowHGT_LR, 'low', 'none'), (FIG3_nINT_highHGT_LR, 'high', 'none'), 
                      (FIG3_wINT_nHGT_LR, 'null', 'yes'), (FIG3_wINT_lowHGT_LR, 'low', 'yes'), (FIG3_wINT_highHGT_LR, 'high', 'yes'))

# ------------------------------------- PLOTTING ----------------------------------------------------------
a_Label = 'Antibiotic Exposure Prevalence (a)'
Prevalence_Label = 'Colonization prevalence (C^R)'
Incidence_Label = 'Daily Colonization Incidence (% of Patients)'
R_Rate_Label = 'Resistance Rate (C^R / C^S + C^R)'

fig, ax = plt.subplots(1, 3) 

ax[0].plot(FIG3_nINT_nHGT_LR['Parameter_Value'], FIG3_nINT_nHGT_LR['Prevalence_CR'], linestyle = '-', color = 'orange')
ax[0].plot(FIG3_nINT_lowHGT_LR['Parameter_Value'], FIG3_nINT_lowHGT_LR['Prevalence_CR'], linestyle = '--', color = 'orange')
ax[0].plot(FIG3_nINT_highHGT_LR['Parameter_Value'], FIG3_nINT_highHGT_LR['Prevalence_CR'], linestyle = ':', color = 'orange')

ax[0].plot(FIG3_wINT_nHGT_LR['Parameter_Value'], FIG3_wINT_nHGT_LR['Prevalence_CR'], linestyle = '-', color = 'purple')
ax[0].plot(FIG3_wINT_lowHGT_LR['Parameter_Value'], FIG3_wINT_lowHGT_LR['Prevalence_CR'], linestyle = '--', color = 'purple')
ax[0].plot(FIG3_wINT_highHGT_LR['Parameter_Value'], FIG3_wINT_highHGT_LR['Prevalence_CR'], linestyle = ':', color = 'purple')
ax[0].set_title('Low Resistance Level (rR = 0.2)')

ax[1].plot(FIG3_nINT_nHGT_MR['Parameter_Value'], FIG3_nINT_nHGT_MR['Prevalence_CR'], linestyle = '-', color = 'orange')
ax[1].plot(FIG3_nINT_lowHGT_MR['Parameter_Value'], FIG3_nINT_lowHGT_MR['Prevalence_CR'], linestyle = '--', color = 'orange')
ax[1].plot(FIG3_nINT_highHGT_MR['Parameter_Value'], FIG3_nINT_highHGT_MR['Prevalence_CR'], linestyle = ':', color = 'orange')

ax[1].plot(FIG3_wINT_nHGT_MR['Parameter_Value'], FIG3_wINT_nHGT_MR['Prevalence_CR'], linestyle = '-', color = 'purple')
ax[1].plot(FIG3_wINT_lowHGT_MR['Parameter_Value'], FIG3_wINT_lowHGT_MR['Prevalence_CR'], linestyle = '--', color = 'purple')
ax[1].plot(FIG3_wINT_highHGT_MR['Parameter_Value'], FIG3_wINT_highHGT_MR['Prevalence_CR'], linestyle = ':', color = 'purple')
ax[1].set_title('Medium Resistance Level (rR = 0.5)')

ax[2].plot(FIG3_nINT_nHGT_HR['Parameter_Value'], FIG3_nINT_nHGT_HR['Prevalence_CR'], linestyle = '-', color = 'orange')
ax[2].plot(FIG3_nINT_lowHGT_HR['Parameter_Value'], FIG3_nINT_lowHGT_HR['Prevalence_CR'], linestyle = '--', color = 'orange')
ax[2].plot(FIG3_nINT_highHGT_HR['Parameter_Value'], FIG3_nINT_highHGT_HR['Prevalence_CR'], linestyle = ':', color = 'orange')

ax[2].plot(FIG3_wINT_nHGT_HR['Parameter_Value'], FIG3_wINT_nHGT_HR['Prevalence_CR'], linestyle = '-', color = 'purple')
ax[2].plot(FIG3_wINT_lowHGT_HR['Parameter_Value'], FIG3_wINT_lowHGT_HR['Prevalence_CR'], linestyle = '--', color = 'purple')
ax[2].plot(FIG3_wINT_highHGT_HR['Parameter_Value'], FIG3_wINT_highHGT_HR['Prevalence_CR'], linestyle = ':', color = 'purple')
ax[2].set_title('High Resistance Level (rR = 0.8)')


for axis in ax:
    axis.set(xlabel = a_Label, ylabel = Prevalence_Label)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 0.5)
    axis.label_outer()

plt.show()

fig, ax = plt.subplots(1, 3) 

ax[0].plot(FIG3_nINT_nHGT_LR['Parameter_Value'], FIG3_nINT_nHGT_LR['R_Rate'], linestyle = '-', color = 'orange')
ax[0].plot(FIG3_nINT_lowHGT_LR['Parameter_Value'], FIG3_nINT_lowHGT_LR['R_Rate'], linestyle = '--', color = 'orange')
ax[0].plot(FIG3_nINT_highHGT_LR['Parameter_Value'], FIG3_nINT_highHGT_LR['R_Rate'], linestyle = ':', color = 'orange')

ax[0].plot(FIG3_wINT_nHGT_LR['Parameter_Value'], FIG3_wINT_nHGT_LR['R_Rate'], linestyle = '-', color = 'purple')
ax[0].plot(FIG3_wINT_lowHGT_LR['Parameter_Value'], FIG3_wINT_lowHGT_LR['R_Rate'], linestyle = '--', color = 'purple')
ax[0].plot(FIG3_wINT_highHGT_LR['Parameter_Value'], FIG3_wINT_highHGT_LR['R_Rate'], linestyle = ':', color = 'purple')
ax[0].set_title('Low Resistance Level (rR = 0.2)')

ax[1].plot(FIG3_nINT_nHGT_MR['Parameter_Value'], FIG3_nINT_nHGT_MR['R_Rate'], linestyle = '-', color = 'orange')
ax[1].plot(FIG3_nINT_lowHGT_MR['Parameter_Value'], FIG3_nINT_lowHGT_MR['R_Rate'], linestyle = '--', color = 'orange')
ax[1].plot(FIG3_nINT_highHGT_MR['Parameter_Value'], FIG3_nINT_highHGT_MR['R_Rate'], linestyle = ':', color = 'orange')

ax[1].plot(FIG3_wINT_nHGT_MR['Parameter_Value'], FIG3_wINT_nHGT_MR['R_Rate'], linestyle = '-', color = 'purple')
ax[1].plot(FIG3_wINT_lowHGT_MR['Parameter_Value'], FIG3_wINT_lowHGT_MR['R_Rate'], linestyle = '--', color = 'purple')
ax[1].plot(FIG3_wINT_highHGT_MR['Parameter_Value'], FIG3_wINT_highHGT_MR['R_Rate'], linestyle = ':', color = 'purple')
ax[1].set_title('Medium Resistance Level (rR = 0.5)')

ax[2].plot(FIG3_nINT_nHGT_HR['Parameter_Value'], FIG3_nINT_nHGT_HR['R_Rate'], linestyle = '-', color = 'orange')
ax[2].plot(FIG3_nINT_lowHGT_HR['Parameter_Value'], FIG3_nINT_lowHGT_HR['R_Rate'], linestyle = '--', color = 'orange')
ax[2].plot(FIG3_nINT_highHGT_HR['Parameter_Value'], FIG3_nINT_highHGT_HR['R_Rate'], linestyle = ':', color = 'orange')

ax[2].plot(FIG3_wINT_nHGT_HR['Parameter_Value'], FIG3_wINT_nHGT_HR['R_Rate'], linestyle = '-', color = 'purple')
ax[2].plot(FIG3_wINT_lowHGT_HR['Parameter_Value'], FIG3_wINT_lowHGT_HR['R_Rate'], linestyle = '--', color = 'purple')
ax[2].plot(FIG3_wINT_highHGT_HR['Parameter_Value'], FIG3_wINT_highHGT_HR['R_Rate'], linestyle = ':', color = 'purple')
ax[2].set_title('High Resistance Level (rR = 0.8)')

for axis in ax:
    axis.set(xlabel = a_Label, ylabel = R_Rate_Label)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.label_outer()


plt.show()
