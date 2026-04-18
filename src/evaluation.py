import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# --------------------------------
# Basic Evaluation Metrics
# --------------------------------

def evaluate_model(y_true, y_pred, y_prob=None):
    """
    Compute evaluation metrics for a model.
    """

    results = {}

    results["accuracy"] = accuracy_score(y_true, y_pred)

    results["precision"] = precision_score(y_true, y_pred)

    results["recall"] = recall_score(y_true, y_pred)

    results["f1_score"] = f1_score(y_true, y_pred)

    if y_prob is not None:
        results["auc"] = roc_auc_score(y_true, y_prob)

    else:
        results["auc"] = None

    return results


# --------------------------------
# Build Results Table
# --------------------------------

def create_results_table(results_dict):
    """
    Convert multiple model results into a dataframe.
    """

    df = pd.DataFrame(results_dict).T

    df.index.name = "Model"

    return df.reset_index()


# --------------------------------
# Plot Model Comparison
# --------------------------------

def plot_model_comparison(results_df, metric="f1_score"):
    """
    Plot comparison between models.
    """

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=results_df,
        x="Model",
        y=metric
    )

    plt.title(f"Model Comparison ({metric})")

    plt.xticks(rotation=30)

    plt.tight_layout()

    plt.show()


# --------------------------------
# Save Results
# --------------------------------

def save_results(results_df, path="results.csv"):
    """
    Save results table to CSV.
    """

    results_df.to_csv(path, index=False)
