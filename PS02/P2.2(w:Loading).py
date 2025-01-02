
# Name : Nery Matias Calmo 
# BME143 : Biological Systems Analysis 
# Problem Set #2 - Problem #2 : Principal Component Analysis (PCA)

import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA 

# Load data set (.mat) file 
Mat_File = loadmat('pset2.mat')
myData = Mat_File['myData'] # Take out the mtrix data
Sign_Names = ['A', 'B', 'C', 'D', 'E']

# Function that takes in a data set and the number of desired components
def PCAnalysis(Data, Components): 
    pca = PCA(n_components = Components) 
    PCA_Data = pca.fit_transform(Data)
    Explained_Variance = pca.explained_variance_ratio_ # Get explained variance ratio from model
    Principal_Components = pca.components_ # Get the principal components 
    return PCA_Data, Explained_Variance, Principal_Components

PCA_Data, EV, PC = PCAnalysis(myData, 5) # 5 signaling entities (A, B, C, D, E)

# Plot the PCA Results and visulaize 
CV = np.cumsum(EV) # Evalulate the Cumulative Variance 
plt.plot(range(1, len(CV) + 1), CV, '-bo', label = 'Cumulative Variance', color = 'darkblue')
plt.bar(range(1, 6), EV, label = 'Component Variance', color = 'darkcyan') # Explained Variance 
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance')
plt.title('Explained Variance by Different Principal Components')
plt.legend()
plt.grid(True)
plt.show()

# Plot the first two principal components 
plt.scatter(PCA_Data[:, 0], PCA_Data[:, 1])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Signalling Entities')
plt.grid(True)
plt.show()

# Plot the loadings with arrows indicating variable contributions
Loadings = PC.T * np.sqrt(EV)
plt.scatter(Loadings[:, 0], Loadings[:, 1])  # Plot loadings for PC1 vs PC2
for i in range(Loadings.shape[0]):
    plt.arrow(0, 0, Loadings[i, 0], Loadings[i, 1], color = 'teal', alpha=0.5)
    plt.text(Loadings[i, 0] * 1.05, Loadings[i, 1] * 1.05, Sign_Names[i], color = 'darkblue', ha = 'center')
plt.xlabel('PC1 Loadings')
plt.ylabel('PC2 Loadings')
plt.title('PCA Loadings Plot')
plt.grid(True)
plt.show()

print('Principal Components: ')
for i, j in enumerate(PC): 
    print (f'\n PC {i + 1}:', j, '\n')