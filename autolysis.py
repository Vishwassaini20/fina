# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
#     "pandas",
#     "requests",
#     "seaborn"
# ]
# ///

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# Set your proxy endpoint
API_PROXY_ENDPOINT = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
GPT_MODEL = "gpt-4o-mini"


def readAPITokenFromEnv():
    """Read environment variable, if it does not exist then raise error"""
    token = os.getenv("AIPROXY_TOKEN")
    if not token:
        raise EnvironmentError("AIPROXY_TOKEN variable not found")
    return token


def inkokeLLMModel(messages, api_token):
    """Invoke LLM Model with the passed messages."""
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "model": GPT_MODEL,
        "messages": messages,
        "temperature": 0.7,
    }
    response = requests.post(API_PROXY_ENDPOINT, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        response.raise_for_status()


def saveChartToDisk(fig, chartName):
    """Persist the passed chart onto disk."""
    pathOnDisk = f"{chartName}.png"
    fig.savefig(pathOnDisk, format="png", bbox_inches="tight")
    plt.close(fig)
    return pathOnDisk


def performPrimitiveAnalysis(data):
    """This method generates primitive summary and returns a dictionary containing the results"""
    results = {}

    # Describe the data
    desriptiveStats = data.describe(include="all").to_dict()
    results["summary_statistics"] = desriptiveStats

    # Find Missing values
    missingValues = data.isnull().sum().to_dict()
    results["missing_values"] = missingValues

    # Correl
    numericalDataColumns = data.select_dtypes(include=["number"])
    if not numericalDataColumns.empty:
        correlation_matrix = numericalDataColumns.corr()
        results["correlations"] = correlation_matrix.to_dict()

        # Visualize correlation matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        saveChartToDisk(fig, "correlation_matrix")

    return results


def visualizeSuggestions(data, suggestions):
    """This method iterates through the LLM provided suggestions, plots the chart and saves them on disk"""
    numericalDataColumns = data.select_dtypes(include=["number"])
    chart_paths = []

    for suggestion in suggestions.split("\n"):
        suggestion = suggestion.lower()

        if "histogram" in suggestion:
            for col in numericalDataColumns.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                numericalDataColumns[col].hist(bins=20, ax=ax)
                ax.set_title(f"Histogram of {col}")
                chart_paths.append(saveChartToDisk(fig, f"histogram_{col}"))

        elif "scatter" in suggestion and numericalDataColumns.shape[1] >= 2:
            cols = numericalDataColumns.columns[:2]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=numericalDataColumns[cols[0]], y=numericalDataColumns[cols[1]], ax=ax)
            ax.set_title(f"Scatterplot of {cols[0]} vs {cols[1]}")
            chart_paths.append(saveChartToDisk(fig, "scatterplot"))

    return chart_paths


def main(filename):
    """This is the orchestration function to weave the workflow"""
    
    # Step 1: Load the token from environment
    api_token = readAPITokenFromEnv()

    # Step 2: Read the dataset into pandas dataframe considering different encodings
    try:
        data = pd.read_csv(filename, encoding='utf-8')  # Try 'utf-8' first
    except UnicodeDecodeError:
        print("UnicodeDecodeError with 'utf-8', trying 'cp1252'")
        try:
            data = pd.read_csv(filename, encoding='cp1252')  # Try 'cp1252' if 'utf-8' fails
        except Exception as e:
            print(f"Error loading CSV file with 'cp1252': {e}")
            return
    except Exception as e:
        print(f"Error loading CSV file with 'utf-8': {e}")
        return

    # Step 3: Perform primitive analysis to be sent to LLM
    analysis_results = performPrimitiveAnalysis(data)

    # Step 4: Prepare the prompt user content for suggestions on visualizations
    summary = {
        "columns": list(data.columns),
        "dtypes": data.dtypes.astype(str).to_dict(),
        "analysis_results": analysis_results,
    }
    summary_text = json.dumps(summary)

    # Step 5: Invoke LLM with the prompt
    suggestions_prompt = [
        {"role": "system", "content": "You are an expert data analyst and visualization specialist."},
        {"role": "user", "content": f"Here is a summary of the dataset:\n{summary_text}\n" +
                                     "Suggest additional analysis or visualizations."}
    ]
    suggestions = inkokeLLMModel(suggestions_prompt, api_token)

    # Step 6: Visualize the suggestions
    additional_charts = visualizeSuggestions(data, suggestions)

    # Step 7: Invoke the LLM with additional details including visualizations and get the narrative
    narrative_prompt = [
        {"role": "system", "content": "You are an AI assistant creating a story based on data analysis."},
        {"role": "user", "content": f"Here is the dataset name, analysis summary, initial insights, and additional charts:\n" +
                                    f"Dataset Name: {filename}\n" +
                                     f"Analysis Summary: {analysis_results}\n" +
                                     f"Charts: {additional_charts}\n" +
                                     "Generate a detailed README.md narrative which should includes 1. Dataset you received 2. The analysis you carried out 3. Include the charts in the markdown file 4. The insights you discovered 5. The implications of your findings (i.e. what to do with the insights)"}
    ]
    narrative = inkokeLLMModel(narrative_prompt, api_token)

    #Step 8: Write the README.md file on the disk
    with open("README.md", "w") as readme_file:
        readme_file.write(narrative)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py nameofthedataset.csv")
        sys.exit(1)

    main(sys.argv[1])
