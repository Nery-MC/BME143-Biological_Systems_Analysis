
# Name : Nery Matias Calmo 
# BME143 : Biological Systems Analysis 
# Problem Set #2 - Problem #3 : Partial Least Square Regression (PLSR)

from nipals import nipals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Import and load the data 
Raw_Data = pd.read_csv('PLSRdata.csv', header = None)

Colors = ['firebrick', 'goldenrod', 'olivedrab', 'teal', 'mediumslateblue']
Time_Points = ['12 h', '24 h', '48 h']

# --------------------------------------------------------------------------
# PART A : create the X and Y matrix and scale using Unit Variance Scaling 
Xraw = Raw_Data.iloc[0:4, 0:40] # Extract Input (Protein Measurements)
Yraw = Raw_Data.iloc[0:4, 40:43] # Extract Output (Death Measurements)

X_scaler = StandardScaler() # Scaler object for X
Xscaled = X_scaler.fit_transform(Xraw) # Fit sacler into X matrix data

Y_scaler = StandardScaler() # Scaler object for Y
Yscaled = Y_scaler.fit_transform(Yraw) # Fit scaler into Y matrix data 

# --------------------------------------------------------------------------
# PART B : use X and Y to generate a 2-component PLSR model 
[P, Q, W, B] = nipals(Xscaled, Yscaled, 2) # P = PC(x), Q = PC(y), W = weight(x), B = regression coeff

# --------------------------------------------------------------------------
# PART C : Plot the predicted (model) output (unscaled) compared to the original output
Ypred = Xscaled @ W @ B @ Q.T # Create a new predicted unscaled Y matrix 
Ypred = Y_scaler.inverse_transform(Ypred) # Unscale the Y matrix 

plt.figure(figsize=(10, 6))

for i in range(Yraw.shape[0]):  # Loop through each sample
    plt.plot([0, 1, 2], Yraw.iloc[i, :], '-o', color=Colors[i], label=f'Sample {i+1} Actual')  # Actual values
    plt.plot([0, 1, 2], Ypred[i, :], '--s', color=Colors[i], label=f'Sample {i+1} Predicted')  # Predicted values

plt.xticks([0, 1, 2], Time_Points)
plt.xlabel('Time')
plt.ylabel('Cell Death %')
plt.title('Comparison of Predictive Ability')
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------------------------------
# PART D : evalulate the X and Y loading and plot both 

# Plot the Y loadings (Q)
width = 0.35 # Width of the bars (for aesthetics)
x = np.arange(Q.shape[0]) 

plt.bar(x - width/2, Q[:, 0], width, color = 'teal', label = 'PC1') 
plt.bar(x + width/2, Q[:, 1], width, color = 'lightblue', label = 'PC2')
plt.xticks(np.arange(Q.shape[0]), ['12HR', '24 HR', '48HR'])
plt.legend()
plt.title('Y Loadings')
plt.grid(True)
plt.show()

# Plot the X loadings (P)
plt.plot(P[:, 0], color = 'slateblue', label='PC1') 
plt.plot(P[:, 1], color = 'goldenrod', label='PC2')  
plt.xlabel('Protein Measurements')
plt.ylabel('Loading Value')
plt.legend()
plt.title('X Loadings')
plt.grid(True)
plt.show()

# --------------------------------------------------------------------------
# PART E : include 5th row into model and rerun 
Xraw = Raw_Data.iloc[0:5, 0:40] # Extract new Input w/ Row 5
Yraw = Raw_Data.iloc[0:5, 40:43] # Extract new Output w/ Row 5
Xscaled = X_scaler.fit_transform(Xraw)
Yscaled = Y_scaler.fit_transform(Yraw)

[P, Q, W, B] = nipals(Xscaled, Yscaled, 2)
Ypred = Xscaled @ W @ B @ Q.T
Ypred = Y_scaler.inverse_transform(Ypred)

plt.figure(figsize=(10, 6))

for i in range(Yraw.shape[0]):  # Loop through each sample
    plt.plot([0, 1, 2], Yraw.iloc[i, :], '-o', color=Colors[i], label=f'Sample {i+1} Actual')
    plt.plot([0, 1, 2], Ypred[i, :], '--s', color=Colors[i], label=f'Sample {i+1} Predicted')

plt.xticks([0, 1, 2], Time_Points)
plt.xlabel('Time')
plt.ylabel('Cell Death %')
plt.title('Comparison of Predictive Ability w/ Sample 5')
plt.legend()
plt.grid(True)
plt.show()

# Plot the Y loadings (Q)
width = 0.35
x = np.arange(Q.shape[0]) 

plt.bar(x - width/2, Q[:, 0], width, color = 'teal', label = 'PC1') 
plt.bar(x + width/2, Q[:, 1], width, color = 'lightblue', label = 'PC2')
plt.xticks(np.arange(Q.shape[0]), ['12HR', '24 HR', '48HR'])
plt.legend()
plt.title('Y Loadings w/ Sample 5')
plt.grid(True)
plt.show()

# Plot the X loadings (P)
plt.plot(P[:, 0], color = 'slateblue', label='PC1') 
plt.plot(P[:, 1], color = 'goldenrod', label='PC2')  
plt.xlabel('Protein Measurements')
plt.ylabel('Loading Value')
plt.legend()
plt.title('X Loadings w/ Sample 5')
plt.grid(True)
plt.show()