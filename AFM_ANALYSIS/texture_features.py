import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

import os
import pandas as pd

class TextureFeatureExtractor:
    """
    A class to extract texture features from an AFM image, including
    Gray-Level Co-Occurrence Matrix (GLCM) features and Local Binary Pattern (LBP) metrics.
    """

    def __init__(self, levels=256):
        """
        Initializes the TextureFeatureExtractor.

        Parameters:
            levels (int): Number of gray levels for GLCM. Defaults to 256.
        """
        self.levels = levels

    def load_afm_txt(self, file_path, output_size=(256, 256)):
        """
        Loads AFM image data from a TXT file containing (x, y, z) values and
        reshapes it into a 2D matrix.

        Parameters:
            file_path (str): Path to the TXT file.
            output_size (tuple): Desired output size as (height, width). Defaults to (256, 256).

        Returns:
            ndarray: 2D image matrix (z-values interpolated to a grid).
        """
        # Load data
        data = np.loadtxt(file_path, delimiter = "\t", skiprows=7,usecols=(0,1,2))

        x, y, z = data[:, 0], data[:, 1], data[:, 2]

        # Define the output grid
        new_x = np.linspace(x.min(), x.max(), output_size[1])
        new_y = np.linspace(y.min(), y.max(), output_size[0])
        new_x_grid, new_y_grid = np.meshgrid(new_x, new_y)

        # Interpolate z-values onto the grid
        points = np.array([x, y]).T
        z_resized = griddata(points, z, (new_x_grid, new_y_grid), method='linear')

        # Handle NaNs resulting from interpolation
        z_resized = np.nan_to_num(z_resized, nan=0.0)

        return z_resized

    def visualize_z_data(self, z_matrix, title = 'AFM Z-Data Visualization', colormap='afmhot', figsize=(8, 6)):
        """
        Visualizes the Z-data matrix as a 2D heatmap.

        Parameters:
            z_matrix (ndarray): 2D array representing Z-data (e.g., height values).
            colormap (str): Matplotlib colormap to use. Defaults to 'viridis'.
            show_colorbar (bool): Whether to show a colorbar. Defaults to True.
            figsize (tuple): Size of the figure as (width, height). Defaults to (8, 6).

        Returns:
            None: Displays the heatmap plot.
        """
        plt.figure(figsize=figsize)
        plt.imshow(z_matrix, cmap=colormap, interpolation = 'sinc', origin='lower')
        plt.xticks([])
        plt.yticks([])
#         plt.imshow(z_matrix, origin='lower')

        plt.title(title,size = 30)
#         plt.xlabel('X-axis (pixels)',size = 30)
#         plt.ylabel('Y-axis (pixels)',size = 30)
        
        
        cbar = plt.colorbar()
        cbar.set_label('Height [nm]',size=30)
        cbar.ax.tick_params(labelsize=15)
        
        plt.show()

    def extract_glcm_features(self, image, distances, angles, props):
        """
        Extracts GLCM features from the image for each distance and angle combination,
        and computes the mean for each property across all combinations.

        Parameters:
            image (ndarray): 2D grayscale image (e.g., 256x256 matrix).
            distances (list): List of pixel pair distance offsets.
            angles (list): List of pixel pair angles in radians.
            props (list): List of properties to extract (e.g., 'contrast', 'homogeneity').

        Returns:
            dict: GLCM properties for each distance and angle combination, along with their means.
        """
        # Convert to uint8 if the image is in float format
        if image.dtype.kind == 'f':  # Check if the image is floating-point
            image = (image - image.min()) / (image.max() - image.min()) * 255
            image = image.astype(np.uint8)

        # Compute GLCM
        glcm = graycomatrix(image, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

        # Store features for each combination
        glcm_features = {}
        property_sums = {prop: 0 for prop in props}
        property_counts = len(distances) * len(angles)

        for i, d in enumerate(distances):
            for j, a in enumerate(angles):
                # Extract features for this distance and angle
                key = f"dist_{d}_angle_{np.degrees(a):.1f}"  # e.g., "dist_1_angle_0.0"
                glcm_features[key] = {}
                for prop in props:
                    value = graycoprops(glcm, prop)[i, j]
                    glcm_features[key][prop] = value
                    property_sums[prop] += value

        # Add mean properties
        glcm_features['mean'] = {prop: property_sums[prop] / property_counts for prop in props}

        return glcm_features



    def extract_lbp_features(self, image, radius, n_points, method='uniform'):
        """
        Extracts LBP features from the image.

        Parameters:
            image (ndarray): 2D grayscale image (e.g., 256x256 matrix).
            radius (int): Radius of the LBP neighborhood.
            n_points (int): Number of sampling points in the LBP neighborhood.
            method (str): Method to compute LBP (e.g., 'uniform'). Defaults to 'uniform'.

        Returns:
            ndarray: Normalized histogram of LBP values.
        """
        self.lbp = local_binary_pattern(image, n_points, radius, method)
        n_bins = int(self.lbp.max() + 1)
        hist, _ = np.histogram(self.lbp, bins=n_bins, range=(0, n_bins), density=True)
        return hist
    
    def process_afm_file(self, file_path, glcm_params=None, lbp_params=None):
        """
        Processes an AFM TXT file to extract GLCM and LBP features.

        Parameters:
            file_path (str): Path to the AFM TXT file.
            glcm_params (dict): Parameters for GLCM extraction.
            lbp_params (dict): Parameters for LBP extraction.

        Returns:
            dict: Flattened dictionary of extracted GLCM and LBP features.
        """
        # Load the image from the TXT file
        image = self.load_afm_txt(file_path)

        # Ensure the image is in uint8 format for GLCM
        if image.dtype.kind == 'f':  # Check if the image is floating-point
            image = (image - image.min()) / (image.max() - image.min()) * 255  # Normalize to [0, 255]
            image = image.astype(np.uint8)

        # Initialize a dictionary for extracted features
        features = {}

        # Extract GLCM features
        if glcm_params:
            features['glcm'] = self.extract_glcm_features(
                image=image,
                distances=glcm_params.get('distances', [1]),
                angles=glcm_params.get('angles', [0]),
                props=glcm_params.get('props', ['contrast', 'homogeneity', 'energy', 'correlation'])
            )

        # Extract LBP features
        if lbp_params:
            lbp_histogram = self.extract_lbp_features(
                image=image,
                radius=lbp_params.get('radius', 1),
                n_points=lbp_params.get('n_points', 8),
                method=lbp_params.get('method', 'uniform')
            )
            features['lbp'] = {f"bin_{i}": val for i, val in enumerate(lbp_histogram)}

        # Flatten GLCM and LBP features for ease of use
        flat_features = {}
        if 'glcm' in features:
            # Flatten GLCM features
            for key, metrics in features['glcm'].items():
                if key == 'mean':
                    for prop, value in metrics.items():
                        flat_features[f"glcm_mean_{prop}"] = value
                else:
                    for prop, value in metrics.items():
                        flat_features[f"glcm_{key}_{prop}"] = value

        if 'lbp' in features:
            # Add LBP histogram bins
            flat_features.update(features['lbp'])

        return flat_features


    
    def visualize_lbp_image(self, image, radius, n_points, method='uniform', figsize=(8, 6)):
        """
        Visualizes the Local Binary Pattern (LBP) image.

        Parameters:
            image (ndarray): 2D grayscale image (e.g., 256x256 matrix).
            radius (int): Radius of the LBP neighborhood.
            n_points (int): Number of sampling points in the LBP neighborhood.
            method (str): Method to compute LBP (e.g., 'uniform'). Defaults to 'uniform'.
            figsize (tuple): Size of the figure as (width, height). Defaults to (8, 6).

        Returns:
            None: Displays the LBP heatmap plot.
        """
        # Compute the LBP image
        lbp_image = local_binary_pattern(image, n_points, radius, method)
        
        lbp_image=self.lbp

        # Plot the LBP image
        plt.figure(figsize=figsize)
        plt.imshow(lbp_image, cmap='gray', origin='lower')
        plt.title('Local Binary Pattern (LBP) Image')
        plt.xlabel('X-axis (pixels)')
        plt.ylabel('Y-axis (pixels)')
        plt.colorbar(label='LBP Value')
        plt.show()

    def extract_features_from_folders(self, folder_path, glcm_params=None, lbp_params=None):
        """
        Extracts features from AFM TXT files located in subfolders of a directory.

        Parameters:
            folder_path (str): Path to the main folder containing subfolders named by temperature.
            glcm_params (dict): Parameters for GLCM extraction.
            lbp_params (dict): Parameters for LBP extraction.

        Returns:
            pd.DataFrame: A dataframe containing features for each file with temperature and file name.
        """
        # Initialize a list to store extracted feature dictionaries
        all_features = []

        # Walk through each subfolder
        for subdir, _, files in os.walk(folder_path):
            # Extract the folder name (used as temperature)
            folder_name = os.path.basename(subdir)

            # Process each file in the current subfolder
            for file in files:
                if file.endswith('.txt'):  # Ensure only TXT files are processed
                    file_path = os.path.join(subdir, file)

                    # Extract features from the file
                    features = self.process_afm_file(file_path, glcm_params, lbp_params)

                    # Add temperature and file name information
                    features['temperature'] = folder_name  # Use folder name as temperature
                    features['file_name'] = file

                    # Append to the list
                    all_features.append(features)

        # Convert the list of features into a DataFrame
        df = pd.DataFrame(all_features)
        return df


def extract_features_from_folders2(base_folder, glcm_params=None, lbp_params=None):
    """
    Extracts texture features from AFM TXT files organized by temperature folders.

    Parameters:
        base_folder (str): Path to the base folder containing temperature subfolders.
        glcm_params (dict): Parameters for GLCM feature extraction.
        lbp_params (dict): Parameters for LBP feature extraction.

    Returns:
        pd.DataFrame: DataFrame containing extracted features and temperature labels.
    """
    # Initialize TextureFeatureExtractor
    extractor = TextureFeatureExtractor()

    # List to store results
    data = []

    # Iterate through temperature folders
    for temp_folder in sorted(os.listdir(base_folder)):
        temp_path = os.path.join(base_folder, temp_folder)

        # Ensure folder is a valid temperature directory (numeric)
        if not temp_folder.isdigit():
            continue

        temperature = int(temp_folder)  # Temperature from folder name

        # Process files within the folder
        for file_name in os.listdir(temp_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(temp_path, file_name)

                # Extract features using the TextureFeatureExtractor
                features = extractor.process_afm_file(file_path, glcm_params, lbp_params)

                # Flatten GLCM and LBP features
                flat_features = {f"glcm_{k}": v for k, v in features.get('glcm', {}).items()}
                flat_features.update({f"lbp_{k}": v for k, v in features.get('lbp', {}).items()})

                # Add temperature and file info
                flat_features['temperature'] = temperature
                flat_features['file_name'] = file_name

                # Append to data list
                data.append(flat_features)

    # Convert to DataFrame
    return pd.DataFrame(data)





