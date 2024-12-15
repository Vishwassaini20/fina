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

import os
import sys
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from dotenv import load_dotenv
from io import BytesIO
import chardet
import argparse
import logging
import plotly.express as px
import plotly.io as pio

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables from .env file
load_dotenv()

AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    logging.error("AIPROXY_TOKEN not found in .env file. Please add it.")
    sys.exit(1)

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {AIPROXY_TOKEN}'
}

def get_ai_story(dataset_summary, dataset_info, visualizations):
    """
    Generate a narrative story for dataset analysis using AI.

    Args:
        dataset_summary (dict): Summary statistics of the dataset.
        dataset_info (dict): Dataset metadata and missing values.
        visualizations (dict): Dictionary of saved visualization paths.

    Returns:
        str: Generated narrative or error message.
    """
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    
    prompt = f"""
    Analyze the following dataset and generate a comprehensive story:

    **Dataset Summary**:
    {dataset_summary}

    **Dataset Info**:
    {dataset_info}

    **Visualizations**:
    {visualizations}
    """

    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2000,
        "temperature": 0.7
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error: {e}")
        return "Error: Unable to generate narrative. Please check the AI service."

    return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No narrative generated.")

def load_data(file_path):
    """
    Load a dataset with automatic encoding detection.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        encoding = result['encoding']
        data = pd.read_csv(file_path, encoding=encoding)
        logging.info(f"Data loaded with {encoding} encoding.")
        return data
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        sys.exit(1)

def basic_analysis(data):
    """
    Perform basic analysis on the dataset.

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        dict: Summary statistics, missing values, and column info.
    """
    summary = data.describe(include='all').to_dict()
    missing_values = data.isnull().sum().to_dict()
    column_info = data.dtypes.to_dict()
    return {"summary": summary, "missing_values": missing_values, "column_info": column_info}

def generate_visualization(fig, plot_name):
    """
    Save and return the path of the plot.

    Args:
        fig (matplotlib.figure.Figure or plotly.graph_objects.Figure): Plot figure.
        plot_name (str): Name of the plot file.

    Returns:
        str: Path to the saved plot.
    """
    if isinstance(fig, plt.Figure):
        path = f"{plot_name}.png"
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
    else:
        path = f"{plot_name}.html"
        pio.write_html(fig, path)
    logging.info(f"Plot saved as {path}")
    return path

def generate_correlation_matrix(data):
    numeric_data = data.select_dtypes(include=[np.number])
    if numeric_data.empty:
        logging.warning("No numeric columns for correlation matrix.")
        return None
    corr = numeric_data.corr()
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="coolwarm", title="Correlation Matrix")
    return generate_visualization(fig, "correlation_matrix")

def generate_pca_plot(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.shape[1] < 2:
        logging.warning("Insufficient numeric columns for PCA.")
        return None
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(StandardScaler().fit_transform(numeric_data))
    fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], title="PCA Plot", labels={"x": "PC1", "y": "PC2"})
    return generate_visualization(fig, "pca_plot")

def dbscan_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.empty:
        logging.warning("No numeric data for DBSCAN.")
        return None
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(scaled_data)
    numeric_data['cluster'] = clusters
    fig = px.scatter(numeric_data, x=numeric_data.columns[0], y=numeric_data.columns[1], color='cluster', title="DBSCAN Clustering")
    return generate_visualization(fig, "dbscan_clusters")

def hierarchical_clustering(data):
    numeric_data = data.select_dtypes(include=np.number).dropna()
    if numeric_data.empty:
        logging.warning("No numeric data for hierarchical clustering.")
        return None
    linked = linkage(numeric_data, 'ward')
    fig, ax = plt.subplots(figsize=(12, 8))
    dendrogram(linked, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram", fontsize=16)
    return generate_visualization(fig, "hierarchical_clustering")

def analyze_and_generate_output(file_path):
    data = load_data(file_path)
    analysis = basic_analysis(data)

    image_paths = {
        'correlation_matrix': generate_correlation_matrix(data),
        'pca_plot': generate_pca_plot(data),
        'dbscan_clusters': dbscan_clustering(data),
        'hierarchical_clustering': hierarchical_clustering(data)
    }

    data_info = {
        "filename": file_path,
        "summary": analysis["summary"],
        "missing_values": analysis["missing_values"],
    }

    narrative = get_ai_story(data_info["summary"], data_info["missing_values"], image_paths)
    if not narrative:
        narrative = "Error: Narrative generation failed. Please verify the AI service."

    with open("README.md", "w") as f:
        f.write(f"## Dataset Analysis\n\n{narrative}")
    logging.info("README.md generated.")

    return narrative, image_paths

def main():
    if len(sys.argv) != 2:
        logging.error("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    file_path = sys.argv[1]
    analyze_and_generate_output(file_path)

if __name__ == "__main__":
    main()
