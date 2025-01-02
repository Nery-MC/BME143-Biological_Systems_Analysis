import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nipals

PLSR_data_path = 'PLSRdata.csv'
PLSR_data = pd.read_csv(PLSR_data_path, header=None)
print(f"PLSR_data shape: {PLSR_data.shape}")

# Extract X (first 40 columns)
X = PLSR_data.iloc[0:4, 0:40].values # X for training samples

# Extract Y (next 3 columns)
Y = PLSR_data.iloc[0:4, 40:43].values # Y for training samples

# Verify the shapes of X and Y
print(f"X shape: {X.shape}") # Expected shape: (4, 40) (4 samples, 40 protein measureme
print(f"Y shape: {Y.shape}") # Expected shape: (4, 3) (4 samples, 3 time points of cell

# Scale X and Y using unit variance scaling
X_std = np.std(X, axis=0)
Y_std = np.std(Y, axis=0)
X_scaled = X / X_std
Y_scaled = Y / Y_std
n_components = 2

# Generate the PLSR model
P, Q, W, B = nipals.nipals(X_scaled, Y_scaled, n_components)

# Plot X loadings
plt.figure(figsize=(10, 6))
for i in range(n_components):
    plt.plot(P[:, i], label=f'X Loadings PC{i+1}') # X loadings for PC1, PC2, etc.
plt.title('X Loadings')
plt.xlabel('Variable Index')
plt.ylabel('Loading Value')
plt.legend()
plt.show()
fig, ax = plt.subplots(figsize=(10, 6))
spacing = 0.5 # Define the spacing between PC_1 and PC_2
bar_width = 0.2 # Width of the bars
n_timepoints = Q.shape[0] # Number of Y variables (e.g., 3 timepoints)
indices = np.arange(n_components) * (n_timepoints * bar_width + spacing) # Increase dis

# Colors for each time point (12 hr, 24 hr, 48 hr)
colors = ['b', 'g', 'r']

# Plotting each time point
for i in range(n_timepoints):
    ax.bar(indices + i * bar_width, Q[i, :], bar_width, color=colors[i], label=f'{["12 h", "24 h", "48 h"]}')

# Set the x-axis labels as principal components
ax.set_xticks(indices + bar_width * (n_timepoints / 2))
ax.set_xticklabels([f'PC_{i+1}' for i in range(n_components)])

# Add labels and title, show plot
ax.set_xlabel('Principal Component')
ax.set_ylabel('Loading Value')
ax.set_title('Y Loadings (Grouped by PC)')
plt.show()

# Calculate T (scores)
T = np.dot(X_scaled, W)

# Predict Y_scaled
Y_scaled_pred = np.dot(T, np.dot(B, Q.T))

# Unscale the predicted Y values
Y_pred = Y_scaled_pred * Y_std

# Plot the predicted vs actual Y values
colors = plt.cm.get_cmap('tab10', Y.shape[0])
plt.figure(figsize=(10, 6))
legends = []
for i in range(Y.shape[0]):
    plt.plot(Y[i, :], '-o', color=colors(i), label=f'Sample {i+1} Actual')
    plt.plot(Y_pred[i, :], '--s', color=colors(i), label=f'Sample {i+1} Predicted')
    legends.append(f'Sample {i+1} Actual')
    legends.append(f'Sample {i+1} Predicted')
plt.legend(legends, loc='upper left')
plt.xticks(ticks=[0, 1, 2], labels=['12 hr', '24 hr', '48 hr'])
plt.xlabel('Time')
plt.ylabel('Percent Cell Death')
plt.title('Predicted vs Actual Percent Cell Death for Training Set')
plt.show()

# Sample 5
X_new = PLSR_data.iloc[4, 0:40].values.reshape(1, -1) # Protein data for the 5th sample
Y_new = PLSR_data.iloc[4, 40:43].values.reshape(1, -1) # Actual cell death for the 5th
X_new_scaled = X_new / X_std # Use the same scaling (X_std) as for the training set
Y_scaled_pred_new = np.dot(X_new_scaled, np.dot(W, np.dot(B, Q.T)))
Y_pred_new = Y_scaled_pred_new * Y_std # Unscale using Y's standard deviation from the
plt.figure(figsize=(10, 6))
plt.plot(Y_new[0], '-o', label='Actual', color='b') # Plot actual Y
plt.plot(Y_pred_new[0], '--s', label='Predicted', color='b') # Plot predicted Y
plt.xticks(ticks=[0, 1, 2], labels=['12 hr', '24 hr', '48 hr']) # Time points for the x
plt.xlabel('Time')
plt.ylabel('Percent Cell Death')
plt.title('Test Set: Predicted vs. Actual Percent Cell Death for Fifth Sample')
plt.legend()
plt.show()


# -------------------------------------- EXAM 1 -------------------------------------------
from sklearn.decomposition import PCA

# PART [A] : Determine Ideal Number of PCs ----------------------------------------------------------------

X = PLSR_data.iloc[:, 0:40].values # Extract Data 
X_std = np.std(X, axis=0) # Scale using unit variance scaling 
X_scaled = X / X_std

# PCA Function
pca = PCA() 
PCA_Data = pca.fit_transform(X_scaled)
EV = pca.explained_variance_ratio_ # Get explained variance ratio from model
PC = pca.components_ # Get the principal components 

# Plot Explained Variance 
CV = np.cumsum(EV) # Evaluate the Cumulative Variance 

plt.plot(range(1, len(CV) + 1), CV, '-bo', label = 'Cumulative Variance', color = 'darkblue')
plt.bar(range(1, len(EV) + 1), EV, label = 'Component Variance', color = 'darkcyan') # Explained Variance 

plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.legend()
plt.grid(True)
plt.show()

# PART [B] : Determine time points for dimension reduction ----------------------------------------------------------------
Time = ['0m', '5m', '15m', '30m', '60m', '90m', '120m', '240m', '480m', '720m']

# Load manipulated data
PLSR_data_path = 'PLSRdata_Manipulated.csv' #rearranged to be a 10 X 20 matrix every 5th row is a new Protein and every colum is a time point
PLSR_data = pd.read_csv(PLSR_data_path, header=None)

# Preform New PCA
X = PLSR_data.iloc[:, 0:40].values # Extract Data 
X_std = np.std(X, axis=0) # Scale using unit variance scaling 
X_scaled = X / X_std

pca = PCA() # PCA to see if the data rearrangement changes the PCA results (shouldnt but still check)
PCA_Data = pca.fit_transform(X_scaled)
EV = pca.explained_variance_ratio_ 
PC = pca.components_ 

CV = np.cumsum(EV) 

plt.plot(range(1, len(CV) + 1), CV, '-bo', label = 'Cumulative Variance', color = 'darkblue')
plt.bar(range(1, len(EV) + 1), EV, label = 'Component Variance', color = 'darkcyan') 

plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.legend()
plt.grid(True)
plt.show()

Loadings = PC.T * np.sqrt(EV)  # Variable loadings

# Plot Loadings 
Loadings = PC.T * np.sqrt(EV)
plt.scatter(Loadings[:, 0], Loadings[:, 1])  # Plot loadings for PC1 vs PC2
for i in range(Loadings.shape[0]):
    plt.arrow(0, 0, Loadings[i, 0], Loadings[i, 1], color = 'teal', alpha=0.5)
    plt.text(Loadings[i, 0] * 1.05, Loadings[i, 1] * 1.05, Time[i], color = 'darkblue', ha = 'center')
plt.xlabel('PC1 Loadings')
plt.ylabel('PC2 Loadings')
plt.title('PCA Loadings Plot')
plt.grid(True)
plt.show()

# New PLSR ------------------------------------ 
# Load dataset
PLSR_data_path = 'PLSRdata.csv'  # Load the unmanipulated dataset
PLSR_data = pd.read_csv(PLSR_data_path, header=None)

# Specify selected columns (12 columns, not 60)
Selected_Colums = [0, 3, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]

# Extract X and Y (use multiple samples for training)
X_train = PLSR_data.iloc[0:4, Selected_Colums].values  # Training data (rows 0-4)
Y_train = PLSR_data.iloc[0:4, 40:43].values  # Corresponding cell death data

# Standardize X and Y
X_std = np.std(X_train, axis=0)
Y_std = np.std(Y_train, axis=0)
X_scaled = X_train / X_std
Y_scaled = Y_train / Y_std

# Train PLSR model 
n_components = 2
P, Q, W, B = nipals.nipals(X_scaled, Y_scaled, n_components)

# Test on new data (5th sample)
X_new = PLSR_data.iloc[4, Selected_Colums].values.reshape(1, -1)  # Test data (5th sample with 12 columns)
Y_new = PLSR_data.iloc[4, 40:43].values.reshape(1, -1)  # Actual cell death for the 5th sample

# Scale X_new using the same scaling as the training set
X_new_scaled = X_new / X_std 

# Perform prediction
Y_scaled_pred_new = np.dot(X_new_scaled, np.dot(W, np.dot(B, Q.T)))
Y_pred_new = Y_scaled_pred_new * Y_std  # Unscale using Y's standard deviation

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.plot(Y_new[0], '-o', label='Actual', color='b')  # Actual Y
plt.plot(Y_pred_new[0], '--s', label='Predicted', color='b')  # Predicted Y
plt.xticks(ticks=[0, 1, 2], labels=['12 hr', '24 hr', '48 hr'])  # Time points for the x-axis
plt.xlabel('Time')
plt.ylabel('Percent Cell Death')
plt.title('Reduced Set: Predicted vs. Actual Percent Cell Death')
plt.legend()
plt.show()

# PART [C] : find two other metrics -----------------------------------------------
# Load dataset
PLSR_data_path = 'PLSRdata_PConcRatio.csv'  # Load the unmanipulated dataset
PLSR_data = pd.read_csv(PLSR_data_path, header=None)

# Extract X and Y (use multiple samples for training)
X_train = PLSR_data.iloc[0:4, 0:10].values  # Training data (rows 0-4)
Y_train = PLSR_data.iloc[0:4, 10:13].values  # Corresponding cell death data

# Standardize X and Y
X_std = np.std(X_train, axis=0)
Y_std = np.std(Y_train, axis=0)
X_scaled = X_train / X_std
Y_scaled = Y_train / Y_std

# Train PLSR model 
n_components = 2
P, Q, W, B = nipals.nipals(X_scaled, Y_scaled, n_components)

# Test on new data (5th sample)
X_new = PLSR_data.iloc[4, 0:10].values.reshape(1, -1)  # Test data (5th sample with 12 columns)
Y_new = PLSR_data.iloc[4, 10:13].values.reshape(1, -1)  # Actual cell death for the 5th sample

# Scale X_new using the same scaling as the training set
X_new_scaled = X_new / X_std 

# Perform prediction
Y_scaled_pred_new = np.dot(X_new_scaled, np.dot(W, np.dot(B, Q.T)))
Y_pred_new = Y_scaled_pred_new * Y_std  # Unscale using Y's standard deviation

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.plot(Y_new[0], '-o', label='Actual', color='b')  # Actual Y
plt.plot(Y_pred_new[0], '--s', label='Predicted', color='b')  # Predicted Y
plt.xticks(ticks=[0, 1, 2], labels=['12 hr', '24 hr', '48 hr'])  # Time points for the x-axis
plt.xlabel('Time')
plt.ylabel('Percent Cell Death')
plt.title('Protein Rations Set: Predicted vs. Actual Percent Cell Death')
plt.legend()
plt.show()


# Load dataset
PLSR_data_path = 'PLSRdata_PConcAvg.csv'  # Load the unmanipulated dataset
PLSR_data = pd.read_csv(PLSR_data_path, header=None)

# Extract X and Y (use multiple samples for training)
X_train = PLSR_data.iloc[0:4, 0:4].values  # Training data (rows 0-4)
Y_train = PLSR_data.iloc[0:4, 4:7].values  # Corresponding cell death data

# Standardize X and Y
X_std = np.std(X_train, axis=0)
Y_std = np.std(Y_train, axis=0)
X_scaled = X_train / X_std
Y_scaled = Y_train / Y_std

# Train PLSR model 
n_components = 2
P, Q, W, B = nipals.nipals(X_scaled, Y_scaled, n_components)

# Test on new data (5th sample)
X_new = PLSR_data.iloc[4, 0:4].values.reshape(1, -1)  # Test data (5th sample with 12 columns)
Y_new = PLSR_data.iloc[4, 4:7].values.reshape(1, -1)  # Actual cell death for the 5th sample

# Scale X_new using the same scaling as the training set
X_new_scaled = X_new / X_std 

# Perform prediction
Y_scaled_pred_new = np.dot(X_new_scaled, np.dot(W, np.dot(B, Q.T)))
Y_pred_new = Y_scaled_pred_new * Y_std  # Unscale using Y's standard deviation

# Plot predicted vs actual
plt.figure(figsize=(10, 6))
plt.plot(Y_new[0], '-o', label='Actual', color='b')  # Actual Y
plt.plot(Y_pred_new[0], '--s', label='Predicted', color='b')  # Predicted Y
plt.xticks(ticks=[0, 1, 2], labels=['12 hr', '24 hr', '48 hr'])  # Time points for the x-axis
plt.xlabel('Time')
plt.ylabel('Percent Cell Death')
plt.title('Protein Avg Set: Predicted vs. Actual Percent Cell Death')
plt.legend()
plt.show()
