"""
connected_components_analysis.py

This module provides functionality to analyze connected components in AFM maps, visualize them,
and calculate their density.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label
from skimage.color import label2rgb
from PIL import Image
from scipy import ndimage

def analyze_connected_components(image, threshold=None, plot=True):
    """
    Analyzes the connected components in an AFM map, visualizes them, and calculates the density.

    Parameters:
        image (ndarray): 2D AFM image as a NumPy array.
        threshold (float or None): If provided, binarizes the image by applying the threshold 
                                   (e.g., to separate terraces from the substrate).
                                   If None, assumes the image is already binary.
        plot (bool): If True, visualizes the labeled connected components.

    Returns:
        dict: A dictionary containing:
            - 'component_count': Number of connected components.
            - 'component_density': Density of components (count per unit area).
            - 'labeled_image': 2D array of labeled components.
    """
    # Step 1: Threshold the image if a threshold is provided
    if threshold is not None:
        binary_image = image > threshold
    else:
        binary_image = image

    # Step 2: Label connected components
    labeled_image, num_components = label(binary_image, connectivity=2, return_num=True)

    # Step 3: Calculate the density
    area = image.shape[0] * image.shape[1]
    component_density = num_components / area

    # Step 4: Visualize the connected components if requested
    if plot:
        plt.figure(figsize=(10, 8))
        labeled_overlay = label2rgb(labeled_image, image=image, bg_label=0)
        plt.imshow(labeled_overlay, cmap='jet', interpolation='nearest')
        plt.title(f"Connected Components: {num_components}")
        plt.axis('off')
        # plt.colorbar()
        plt.show()

    # Return results
    return {
        'component_count': num_components,
        'component_density': component_density,
        'labeled_image': labeled_image
    }
