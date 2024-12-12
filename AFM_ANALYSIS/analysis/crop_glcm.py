import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

def crop_and_analyze_glcm(image, n, distances, angles, prop):
    """
    Crop a square image into n equal windows, compute GLCM features for each window,
    and plot both the windows and the GLCM features as heatmaps.

    Parameters:
        image (ndarray): A square image as a 2D numpy array.
        n (int): Total number of windows (must be a perfect square).
        distances (list): List of pixel pair distance offsets for GLCM computation.
        angles (list): List of angles (in radians) for GLCM computation.
        prop (str): The GLCM property to compute (e.g., 'contrast', 'energy').

    Returns:
        None
    """
    image_size = image.shape[0]
    if image_size != image.shape[1]:
        raise ValueError("The input image must be square.")

    # Calculate the number of windows per row and the size of each window
    num_windows_per_row = int(np.sqrt(n))
    if num_windows_per_row ** 2 != n:
        raise ValueError("The number of windows (n) must be a perfect square.")

    window_size = image_size // num_windows_per_row

    # Initialize GLCM feature matrix
    glcm_features = np.zeros((num_windows_per_row, num_windows_per_row))

    # Crop the image into n windows and compute GLCM features
    windows = []
    for i in range(num_windows_per_row):
        for j in range(num_windows_per_row):
            x_start = i * window_size
            y_start = j * window_size
            window = image[x_start:x_start + window_size, y_start:y_start + window_size]
            windows.append(window)

            # Normalize window values to integers (required for GLCM)
            window = (window * 255).astype(np.uint8)

            # Compute GLCM and extract the specified feature
            glcm = graycomatrix(window, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            feature_value = graycoprops(glcm, prop=prop)[0, 0]
            glcm_features[i, j] = feature_value

    # Plot the cropped windows
    fig, axes = plt.subplots(num_windows_per_row, num_windows_per_row, figsize=(10, 10))
    for idx, ax in enumerate(axes.ravel()):
        ax.imshow(windows[idx], cmap='gray')
        ax.axis('off')
    plt.suptitle('Cropped Windows')
    plt.tight_layout()
    plt.show()

    # Plot the GLCM features
    plt.figure(figsize=(8, 6))
    plt.imshow(glcm_features, cmap='viridis', origin='upper')
    plt.colorbar(label=f'GLCM {prop.capitalize()}')
    plt.title(f'GLCM {prop.capitalize()} Heatmap')
    plt.xlabel('Window Index (Column)')
    plt.ylabel('Window Index (Row)')
    plt.show()