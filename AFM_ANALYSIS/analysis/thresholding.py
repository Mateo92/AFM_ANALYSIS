import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def determine_threshold_with_gmm(image, max_clusters=5, plot=True):
    """
    Determines an optimal threshold for binarizing an image using Gaussian Mixture Models (GMM).
    
    Parameters:
        image (ndarray): 2D grayscale image.
        max_clusters (int): Maximum number of clusters to consider for the GMM.
        plot (bool): If True, plots the histogram and fitted GMM components.
    
    Returns:
        float: Optimal threshold value.
    """
    # Flatten the image and remove NaN or infinite values
    pixel_values = image.ravel()
    pixel_values = pixel_values[np.isfinite(pixel_values)]
    
    # Compute the histogram
    hist, bin_edges = np.histogram(pixel_values, bins=256, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Prepare data for fitting GMM (repeat bin_centers based on histogram counts)
    weighted_data = np.repeat(bin_centers, (hist * len(pixel_values)).astype(int))

    # Optimize the number of clusters using BIC
    best_gmm = None
    lowest_bic = np.inf
    for n_clusters in range(2, max_clusters + 1):
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm.fit(weighted_data.reshape(-1, 1))
        bic = gmm.bic(weighted_data.reshape(-1, 1))
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm

    # Extract Gaussian means and compute thresholds
    means = np.sort(best_gmm.means_.flatten())
    thresholds = (means[:-1] + means[1:]) / 2  # Midpoints between adjacent means

    # Choose the first threshold (can be adjusted based on use case)
    optimal_threshold = thresholds[0]

    # Plot the histogram and fitted GMM components
    if plot:
        plt.figure(figsize=(10, 6))
        plt.hist(pixel_values, bins=256, density=True, alpha=0.5, label='Histogram')
        x = np.linspace(min(bin_edges), max(bin_edges), 1000).reshape(-1, 1)
        y = np.exp(best_gmm.score_samples(x))
        plt.plot(x, y, label='Fitted GMM', color='red')
        for t in thresholds:
            plt.axvline(t, color='blue', linestyle='--', label=f'Threshold: {t:.2f}')
        plt.title("Histogram and Gaussian Mixture Model")
        plt.xlabel("Pixel Intensity")
        plt.ylabel("Density")
        plt.legend()
        plt.show()

    return optimal_threshold
