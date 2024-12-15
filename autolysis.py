
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "argparse",
#     "matplotlib",
#     "pandas",
#     "requests",
#     "seaborn",
#     "scikit-learn",
#     "missingno",
#     "numpy",
# ]
# ///

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import stats
import missingno as msno
import numpy as np
import time

# Ensure environment variable for AI Proxy Token is set
AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    exit(1)

# Headers for AI Proxy requests
AIPROXY_URL = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {AIPROXY_TOKEN}"
}

def query_llm(messages, temperature=0.7, max_tokens=1000, retries=3):
    """
    Query the LLM via AI Proxy with retry mechanism.
    """
    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    for attempt in range(retries):
        try:
            response = requests.post(AIPROXY_URL, headers=HEADERS, json=payload)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(f"Error querying LLM: {response.status_code}\n{response.text}")
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
        time.sleep(3)  # wait before retry
    print("Error querying LLM after multiple attempts.")
    exit(1)

def analyze_dataset(csv_filename):
    """
    Load the dataset and return basic summary and first few rows for LLM analysis.
    """
    try:
        # Load the dataset
        df = pd.read_csv(csv_filename, encoding="ISO-8859-1")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        exit(1)

    # Preview first few rows for LLM input
    dataset_preview = df.head(20).to_dict()

    # Basic information about the dataset
    summary = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_data": dataset_preview
    }

    return df, summary

def initial_analysis_request(df, data_summary):
    """
    Ask the LLM to perform an initial analysis of the dataset and suggest methods.
    """
    llm_messages = [
        {"role": "system", "content": "You are a data analyst. Please analyze the provided dataset and suggest some appropriate analysis techniques."},
        {"role": "user", "content": f"Here is a preview of the dataset (first 20 rows): {data_summary['sample_data']}. Please provide some insights and recommend charts or analysis techniques that could be useful for this dataset."}
    ]
    initial_insight = query_llm(llm_messages)
    return initial_insight

def perform_analysis_and_visualization(df, initial_insight):
    """
    Perform analysis as suggested by LLM and generate visualizations.
    """
    methods = []
    insights = []

    # Step 1: Apply suggested analysis method based on LLM insights
    if "regression" in initial_insight.lower():
        regression_results = apply_linear_regression(df)
        methods.append("Linear Regression")
        insights.append(regression_results)

    if "clustering" in initial_insight.lower():
        clustering_results = apply_kmeans_clustering(df)
        methods.append("KMeans Clustering")
        insights.append(clustering_results)

    if "pca" in initial_insight.lower():
        pca_results = apply_pca(df)
        methods.append("Principal Component Analysis (PCA)")
        insights.append(pca_results)

    if "time series" in initial_insight.lower():
        time_series_results = apply_time_series_analysis(df)
        methods.append("Time Series Analysis")
        insights.append(time_series_results)

    # Additional advanced statistical methods
    if "hypothesis testing" in initial_insight.lower():
        hypothesis_test_results = apply_hypothesis_testing(df)
        methods.append("Hypothesis Testing")
        insights.append(hypothesis_test_results)

    return methods, insights

def apply_hypothesis_testing(df):
    """
    Apply hypothesis testing (e.g., t-test) and generate a report.
    """
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) >= 2:
        t_stat, p_value = stats.ttest_ind(df[numerical_cols[0]].dropna(), df[numerical_cols[1]].dropna())
        return [f"T-test: {numerical_cols[0]} vs {numerical_cols[1]} - p-value: {p_value}"]
    return []

def apply_time_series_analysis(df):
    """
    Apply time series analysis and generate relevant plot.
    Assuming there is a 'Date' column or similar for time series.
    """
    if 'Date' in df.columns:
        # Convert 'Date' column to datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Drop rows with invalid dates
        
        # Example: Simple line plot for a time series (you can customize this)
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Value'])  # Assuming 'Value' is the time series column
        plt.title('Time Series Analysis')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("time_series_plot.png")
        plt.close()
        return ["time_series_plot.png"]
    return []

def apply_linear_regression(df):
    """
    Apply linear regression and generate relevant plot.
    """
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) >= 2:
        sns.regplot(x=numerical_cols[0], y=numerical_cols[1], data=df, scatter_kws={"color": "red"}, line_kws={"color": "blue"})
        plt.title(f"Linear Regression: {numerical_cols[0]} vs {numerical_cols[1]}")
        plt.xlabel(numerical_cols[0])
        plt.ylabel(numerical_cols[1])
        plt.legend(["Regression Line", "Data Points"])
        plt.savefig("regression_plot.png")
        plt.close()
        return ["regression_plot.png"]
    return []

def apply_kmeans_clustering(df):
    """
    Apply KMeans clustering and generate relevant plot.
    """
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) >= 2:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[numerical_cols[:2]].dropna())
        kmeans = KMeans(n_clusters=3)
        df['Cluster'] = kmeans.fit_predict(scaled_data)
        sns.scatterplot(x=df[numerical_cols[0]], y=df[numerical_cols[1]], hue=df['Cluster'], palette="Set1")
        plt.title(f"KMeans Clustering: {numerical_cols[0]} vs {numerical_cols[1]}")
        plt.xlabel(numerical_cols[0])
        plt.ylabel(numerical_cols[1])
        plt.legend(["Cluster 1", "Cluster 2", "Cluster 3"])
        plt.savefig("kmeans_plot.png")
        plt.close()
        return ["kmeans_plot.png"]
    return []

def apply_pca(df):
    """
    Apply PCA (Principal Component Analysis) and generate relevant plot.
    """
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) >= 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[numerical_cols].dropna())
        df['PCA1'] = pca_result[:, 0]
        df['PCA2'] = pca_result[:, 1]
        sns.scatterplot(x='PCA1', y='PCA2', data=df)
        plt.title("PCA Plot")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend(["Data Points"])
        plt.savefig("pca_plot.png")
        plt.close()
        return ["pca_plot.png"]
    return []

def apply_missing_value_analysis(df):
    """
    Apply missing value analysis and generate relevant plot.
    """
    msno.matrix(df)
    plt.title("Missing Value Heatmap")
    plt.savefig("missing_value_heatmap.png")
    plt.close()
    return ["missing_value_heatmap.png"]

def apply_correlation_matrix(df):
    """
    Apply correlation matrix analysis and generate relevant plot.
    """
    numerical_cols = df.select_dtypes(include="number").columns
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.savefig("correlation_matrix.png")
        plt.close()
        return ["correlation_matrix.png"]
    return []

def generate_readme(data_summary, methods, insights, initial_insight):
    """
    Generate README.md with analysis narrative and references to charts.
    """
    with open("README.md", "w") as f:
        f.write("# Automated Dataset Analysis\n\n")
        f.write("## Dataset Summary\n")
        f.write(f"- Number of Rows: {data_summary['num_rows']}\n")
        f.write(f"- Number of Columns: {data_summary['num_columns']}\n")
        f.write("### Columns and Data Types:\n")
        for col, dtype in data_summary["columns"].items():
            f.write(f"- {col}: {dtype}\n")
        f.write("### Missing Values:\n")
        for col, missing in data_summary["missing_values"].items():
            f.write(f"- {col}: {missing}\n")
        f.write("\n## Initial Insights from LLM\n")
        f.write(f"{initial_insight}\n")
        
        f.write("\n## Analysis Methods\n")
        for method, result in zip(methods, insights):
            f.write(f"- **{method}**: \n")
            for plot in result:
                f.write(f"    ![Image]({plot})\n")

# Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform dataset analysis")
    parser.add_argument("csv_filename", help="Path to the dataset CSV file")
    args = parser.parse_args()

    # Load dataset and analyze
    df, data_summary = analyze_dataset(args.csv_filename)
    initial_insight = initial_analysis_request(df, data_summary)

    # Perform Analysis
    methods, insights = perform_analysis_and_visualization(df, initial_insight)

    # Additional Analysis
    missing_value_plots = apply_missing_value_analysis(df)
    correlation_plots = apply_correlation_matrix(df)

    # Update the insights with additional analysis results
    insights.extend(missing_value_plots + correlation_plots)
    methods.extend(["Missing Value Heatmap", "Correlation Matrix"])

    # Generate and write the analysis summary to README
    generate_readme(data_summary, methods, insights, initial_insight)
