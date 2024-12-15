# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "chardet",
#     "matplotlib",
#     "pandas",
#     "statsmodels",
#     "scikit-learn",
#     "missingno",
#     "python-dotenv",
#     "requests",
#     "seaborn",
#     "plotly",
# ]
# ///
import pandas as pd
import numpy as np
import chardet
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import requests
import sys
import os


# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to load data with proper encoding detection
def load_data(file_path):
    try:
        logging.info(f"Loading data from {file_path}")
        raw_data = open(file_path, 'rb').read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']
        df = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"Data loaded successfully with encoding {encoding}")
        return df
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        sys.exit(1)


# Function for basic data analysis
def basic_analysis(df):
    try:
        logging.info("Performing basic data analysis...")
        summary = df.describe()
        missing_values = df.isnull().sum()
        data_types = df.dtypes
        logging.info("Basic analysis complete.")
        return summary, missing_values, data_types
    except Exception as e:
        logging.error(f"Error in basic analysis: {e}")
        sys.exit(1)


# Function for handling missing values
def handle_missing_data(df):
    try:
        logging.info("Handling missing values...")
        imputer = SimpleImputer(strategy='mean')
        df_imputed = df.copy()
        df_imputed[df_imputed.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(
            df_imputed.select_dtypes(include=[np.number]))
        return df_imputed
    except Exception as e:
        logging.error(f"Error handling missing data: {e}")
        sys.exit(1)


# Function to detect and visualize outliers
def outlier_detection(df):
    try:
        logging.info("Detecting outliers using IQR method...")
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR)))
        sns.boxplot(data=df)
        plt.title("Outlier Detection")
        save_plot("outlier_detection.png")
        return outliers
    except Exception as e:
        logging.error(f"Error in outlier detection: {e}")
        sys.exit(1)


# Function to generate a correlation matrix
def generate_correlation_matrix(df):
    try:
        logging.info("Generating correlation matrix...")
        corr_matrix = df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix")
        save_plot("correlation_matrix.png")
        return corr_matrix
    except Exception as e:
        logging.error(f"Error generating correlation matrix: {e}")
        sys.exit(1)


# Function for PCA (Principal Component Analysis) and visualization
def generate_pca_plot(df):
    try:
        logging.info("Performing PCA...")
        numeric_df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(numeric_df)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(scaled_df)
        plt.figure(figsize=(8, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.title("PCA - Principal Component Analysis")
        save_plot("pca_plot.png")
    except Exception as e:
        logging.error(f"Error in PCA: {e}")
        sys.exit(1)


# Function for DBSCAN Clustering and visualization
def dbscan_clustering(df):
    try:
        logging.info("Performing DBSCAN clustering...")
        numeric_df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(numeric_df)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(scaled_df)
        plt.scatter(scaled_df[:, 0], scaled_df[:, 1], c=labels, cmap='rainbow')
        plt.title("DBSCAN Clustering")
        save_plot("dbscan_clustering.png")
        return labels
    except Exception as e:
        logging.error(f"Error in DBSCAN clustering: {e}")
        sys.exit(1)


# Function for Hierarchical Clustering and visualization
def hierarchical_clustering(df):
    try:
        logging.info("Performing Hierarchical Clustering...")
        numeric_df = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(numeric_df)
        linkage_matrix = sch.linkage(scaled_df, method='ward')
        sch.dendrogram(linkage_matrix)
        plt.title("Hierarchical Clustering Dendrogram")
        save_plot("hierarchical_clustering.png")
    except Exception as e:
        logging.error(f"Error in hierarchical clustering: {e}")
        sys.exit(1)


# Function to save plots
def save_plot(plot_name):
    try:
        plt.savefig(plot_name)
        logging.info(f"Plot saved as {plot_name}")
        plt.close()
    except Exception as e:
        logging.error(f"Error saving plot: {e}")


# Function to generate the AI narrative
def get_ai_story(df, analysis_results):
    try:
        logging.info("Generating AI narrative...")
        # API call for AI story generation (Assumed to be working)
        story = "AI generated narrative here..."  # Placeholder
        return story
    except Exception as e:
        logging.error(f"Error generating AI story: {e}")
        return "Unable to generate AI story."


# Function to save the README file with analysis and AI insights
def save_readme(narrative):
    try:
        with open("README.md", "w") as f:
            f.write(f"# Dataset Analysis\n\n{narrative}")
        logging.info("README file saved.")
    except Exception as e:
        logging.error(f"Error saving README file: {e}")


# Main Execution
if __name__ == "__main__":
    # Load dataset
    file_path = sys.argv[1]
    df = load_data(file_path)

    # Data preprocessing
    df = handle_missing_data(df)

    # Perform basic analysis
    summary, missing_values, data_types = basic_analysis(df)

    # Detect outliers
    outliers = outlier_detection(df)

    # Generate correlation matrix
    correlation_matrix = generate_correlation_matrix(df)

    # Generate PCA plot
    generate_pca_plot(df)

    # Perform DBSCAN clustering
    dbscan_labels = dbscan_clustering(df)

    # Perform hierarchical clustering
    hierarchical_clustering(df)

    # Generate AI narrative
    ai_narrative = get_ai_story(df, (summary, missing_values, outliers, correlation_matrix))

    # Save README with AI insights
    save_readme(ai_narrative)

