# Name : Nery Matias Calmo 
# BME 143 : Biological Systems Analysis 
# Date : December 20, 2024
# Microbiome-Pathogen Interactions Drive Epidemiological Dynamics of ABR

# Final Project Model Implementation : Function.py
# --------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

def Univar_Sweep(i, Values, Model, ODE, State, Parameters): 

    if Model not in ['Model_1', 'Model_2', 'Model_3', 'Model_4', 'Model_5']:
        raise ValueError('Invalid model selection. Choose from Model 1 to Model 5')

    Time = (0, 100)
    Time_Span = np.linspace(0, 100, 1000)

    Results = []
    for value in Values:
        Parameters[i] = value
        try:
            Solution = solve_ivp(ODE, Time, State, args=(Parameters,), t_eval=Time_Span)
            Last_State = Solution.y[:, -1]
                    
            # Calculate model-specific metrics 
            if Model == 'Model_1': 
                CR = Last_State[1]

                Prevalence_CS = 0
                Prevalence_CR = CR
                R_Rate = Prevalence_CR / (Prevalence_CS + Prevalence_CR)
                Incidence_CS = Last_State[2] + Last_State[3]
                Incidence_CR = Last_State[4] + Last_State[5] + Last_State[6]

            elif Model == 'Model_2': 
                CS = Last_State[1]
                CR = Last_State[2]

                Prevalence_CS = CS
                Prevalence_CR = CR
                R_Rate = Prevalence_CR / (Prevalence_CS + Prevalence_CR)
                Incidence_CS = Last_State[3] + Last_State[4]
                Incidence_CR = Last_State[5] + Last_State[6] + Last_State[7]

            elif Model == 'Model_3': 
                CS = 0
                CR = Last_State[2] + Last_State[3]

                Prevalence_CS = CS
                Prevalence_CR = CR
                R_Rate = 1
                Incidence_CS = Last_State[4] + Last_State[5]
                Incidence_CR = Last_State[6] + Last_State[7] + Last_State[8]

            elif Model == 'Model_4':
                CS = Last_State[2] + Last_State[3]
                CR = Last_State[4] + Last_State[5]

                Prevalence_CS = CS
                Prevalence_CR = CR
                R_Rate = Prevalence_CR / (Prevalence_CS + Prevalence_CR)

                Incidence_CS = Last_State[6] + Last_State[7]
                Incidence_CR = Last_State[8] + Last_State[9] + Last_State[10]

            elif Model == 'Model_5':
                CS = Last_State[4] + Last_State[5] + Last_State[6] + Last_State[7]
                CR = Last_State[8] + Last_State[9] + Last_State[10] + Last_State[11]

                Prevalence_CS = CS
                Prevalence_CR = CR
                R_Rate = Prevalence_CR / (Prevalence_CS + Prevalence_CR)

                Incidence_CS = Last_State[12] + Last_State[13]
                Incidence_CR = Last_State[14] + Last_State[15] + Last_State[16]
            
            Results.append({
                'Parameter_Value': value,
                'Prevalence_CS': Prevalence_CS,
                'Prevalence_CR': Prevalence_CR,
                'R_Rate': R_Rate,
                'Incidence_CS': Incidence_CS,
                'Incidence_CR': Incidence_CR
            })
        except Exception as e:
            print(f'ODE integration failed for parameter value {value}: {e}')
            Results.append({
                'Parameter_Value': value,
                'Error': str(e)
            })

    return pd.DataFrame(Results)
    

 

    



