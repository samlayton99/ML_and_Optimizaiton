# Facial Recognition using Eigenfaces

This project employs the foundational technique of eigenfaces for facial recognition, leveraging eigenvectors and Singular Value Decomposition (SVD) to efficiently distinguish between faces. By focusing on eigenfaces, which represent the most significant facial features, we can achieve both computational efficiency and accuracy. 

## Mathematical Overview

The eigenfaces approach for facial recognition employs linear algebra's principles, particularly eigenvectors and Singular Value Decomposition (SVD). Each face in a dataset is viewed as a data point in high-dimensional space, with primary variations captured by eigenvectors. When derived from the facial data's covariance matrix, these become "eigenfaces." These eigenfaces, orthogonal to each other, highlight dominant facial traits like the contours of eyes or nose.

Prior to recognition, images are shifted by their mean, emphasizing distinct features. This involves calculating the "mean face" \( \mu \) by averaging the vectors in dataset \( F \) and then centering each face around this mean. Singular Value Decomposition (SVD) then decomposes facial data into three matrices: \( U \), \( \Sigma \), and \( VT \). Notably, \( U \)'s columns represent the eigenfaces. By focusing on significant singular values, we achieve a condensed representation of faces, accelerating the recognition process without compromising accuracy.

## Key Functionalities

- **display_image()**: This function reshapes and displays flattened image vectors. By converting one-dimensional arrays back to their original 2D matrix form, we can view images in grayscale.

- **FacialRec Class**: At its core, this class manages facial data, providing functionalities such as:
   - Constructing an image database from a given directory and computing the mean face and shifted faces.
   - Projecting input data using the SVD decomposition principle to reduce dimensionality.
   - Matching a given image vector to the closest face in the database.
   - Displaying the original and best-matching face side-by-side for visual comparison.

- **sample_faces() Generator**: This tool fetches random face images from the `faces94` dataset, offering a practical way to test the facial recognition system's robustness.

## Project Flow

1. **Data Visualization**: Start by extracting and displaying sample images to understand the dataset's structure.
2. **Initialize Recognition System**: Set up the core facial recognition system using the `FacialRec` class and visualize transformations like the mean face.
3. **Compute Eigenfaces**: Focus on deriving eigenfaces to represent crucial facial features.
4. **Project & Reconstruct**: Demonstrate the system's efficiency by projecting a face onto eigenfaces and then reconstructing it.
5. **Face Matching**: Test the system by matching random faces against the stored database.

## Required Packages

Ensure these packages are installed for smooth execution:

- Built-in `os` module for OS-related functionalities.
- `numpy` for scientific computations.
- `imageio` for reading/writing image data.
- `matplotlib.pyplot` for plotting and visualizing data.

Installation command:
```bash
pip install numpy imageio matplotlib
```