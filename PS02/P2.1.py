
# Name : Nery Matias Calmo 
# BME143 : Biological Systems Analysis 
# Problem Set #2 - Problem #1 : Single Value Decomposition (SVD) Image Analysis 

import numpy as np 
import matplotlib.pyplot as plt 
from skimage import io, color, img_as_float 

# PART [A] : preform a Singular Value Decomposition (SVD) on an image input 
path = input("Enter Pathname: ") # Prompt user for iamge path 
img = io.imread(path) # Read / load in the image 

# Drop the 'alpha channel for loaded images
if len(img.shape) == 4:
    img = img[0, :, :, :]

# SVD only handles 2 dimensional arrays (ie, grayscale) and put in matrix
img = color.rgb2gray(img)
matrix = img_as_float(img)

print("Shape of image matrix: ", matrix.shape)

# Preform SVD on image 
U, S, VT = np.linalg.svd(matrix, full_matrices = False)

# --------------------------------------------------------------------------
# PART [B] : reconstruct the image w/ different number of S, U, VT
def reconstruct(U, S, VT, k): 
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    VTk = VT[:k, :]

    return Uk @ Sk @ VTk

# Reconstruct images with varying vlaues 
img1 = reconstruct(U, S, VT, 1)
img5 = reconstruct(U, S, VT, 5)
img50 = reconstruct(U, S, VT, 50)

# Plot the images into subplots for comparasion 
fig, ax = plt.subplots(1, 4) # Creates figure w/ 1 row and 4 columns
ax[0].imshow(matrix)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(img1)
ax[1].set_title('1 Singular Value')
ax[1].axis('off')

ax[2].imshow(img5)
ax[2].set_title('5 Singular Values')
ax[2].axis('off')

ax[3].imshow(img50)
ax[3].set_title('50 Singular Vlaues')
ax[3].axis('off')

plt.tight_layout()
plt.show()

# Plot the singular vlaues 
plt.plot(S)
plt.title('Singular Values from SVD')
plt.xlabel('Index [k]')
plt.ylabel('Singular Value [S]k')
plt.grid(True)
plt.show()

img20 = reconstruct(U, S, VT, 20)
plt.imshow(img20)
plt.title('20 Singular Vlaues')
plt.axis('off')
plt.show()