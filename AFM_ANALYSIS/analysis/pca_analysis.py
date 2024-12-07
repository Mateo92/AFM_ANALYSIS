"""
pca_analysis.py

This module provides functions for performing Principal Component Analysis (PCA) 
on extracted features and visualizing the results.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap

def perform_pca_and_plot(df, features, temp_col='temperature', n_components=3):
    """
    Performs PCA on the specified features of a dataframe and plots the first two or three principal components.

    Parameters:
        df (pd.DataFrame): The dataframe containing the data.
        features (list): List of feature column names to include in PCA.
        temp_col (str): Column name indicating the temperature marker.
        n_components (int): Number of principal components to compute.

    Returns:
        pca_results (pd.DataFrame): Dataframe with the PCA results and temperature markers.
    """

    # Ensure the features are numeric
    data = df[features].copy()
    temperatures = df[temp_col].values
    # print(temperatures)

    # Standardize the features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    # print(data_scaled)

    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(data_scaled)


    # Create a dataframe for PCA results
    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(pca_data, columns=pca_columns)
    # print(temperatures)
    # print(pca_df)
    pca_df[temp_col] = temperatures

    # print(pca_df)

    # Assign unique colors to temperature markers

    # unique_markers = pca_df[temp_col].unique()

    unique_markers = df[temp_col].unique()
    # print(unique_markers)

    color_map = ListedColormap(plt.cm.tab10.colors[:len(unique_markers)])
    marker_to_color = {marker: color_map(i) for i, marker in enumerate(unique_markers)}
    # print(marker_to_color)

    # Plot the first two principal components
    plt.figure(figsize=(10, 8))
    for marker, color in marker_to_color.items():
        # print(marker)
        subset = pca_df[pca_df[temp_col] == marker]
        # print(subset)
        plt.scatter(
            subset['PC1'], 
            subset['PC2'], 
            label=str(marker), 
            color=color, 
            edgecolor='k', 
            alpha=0.7
        )
    plt.title("PCA Plot: PC1 vs PC2")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title=temp_col)
    plt.grid(True)
    plt.show()

    # Plot the first three principal components if n_components >= 3
    if n_components >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        for marker, color in marker_to_color.items():
            subset = pca_df[pca_df[temp_col] == marker]
            ax.scatter(
                subset['PC1'], 
                subset['PC2'], 
                subset['PC3'], 
                label=str(marker), 
                color=color, 
                edgecolor='k', 
                alpha=0.7
            )
        ax.set_title("PCA Plot: PC1 vs PC2 vs PC3")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")
        ax.legend(title=temp_col)
        plt.show()

    return pca_df
