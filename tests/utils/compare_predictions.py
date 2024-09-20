import sys
import pandas as pd
from sklearn.metrics import mean_absolute_error

metrics_map = {"mean_absolute_error": mean_absolute_error}


def compare_predictions(
    true_values_path, predictions_path, error_threshold, metric, target_col
):
    try:
        y_true = pd.read_csv(true_values_path)
    except FileNotFoundError:
        print(f"Error: The file {true_values_path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {true_values_path} is empty or not well formatted.")
        sys.exit(1)

    try:
        y_pred = pd.read_csv(predictions_path)
    except FileNotFoundError:
        print(f"Error: The file {predictions_path} was not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file {predictions_path} is empty or not well formatted.")
        sys.exit(1)

    # Verify that both files contain the required 'id' column
    if "id" not in y_true.columns or "id" not in y_pred.columns:
        print("Error: One or both files are missing the 'id' column.")
        sys.exit(1)

    # Merge datasets to ensure they are comparable
    try:
        merged_data = pd.merge(y_true, y_pred, on="id", suffixes=("_true", "_pred"))
    except KeyError:
        print("Error: ID mismatch between the true values and predictions.")
        sys.exit(1)
    # Check if the target column exists and is numeric
    if f"{target_col}_pred" not in merged_data.columns:
        print(f"Error: The target column '{target_col}' does not exist in your file.")
        sys.exit(1)

    # Ensure the target column contains numeric data for both true and predicted values
    if not pd.api.types.is_numeric_dtype(
        merged_data[f"{target_col}_true"]
    ) or not pd.api.types.is_numeric_dtype(merged_data[f"{target_col}_pred"]):
        print(
            f"Error: The target column '{target_col}' must contain numeric data in both true values and predictions."
        )
        sys.exit(1)

    try:
        score = metrics_map[metric](
            merged_data[f"{target_col}_true"], merged_data[f"{target_col}_pred"]
        )
    except KeyError:
        print(f"Error: The metric '{metric}' is not supported.")
        sys.exit(1)

    # Check if the score exceeds the threshold
    if score > error_threshold:
        print(f"Error: {metric} score: {score} exceeds threshold {error_threshold}.")
        sys.exit(1)
    else:
        print(f"Success: {metric} score: {score} is within the acceptable threshold.")


if __name__ == "__main__":
    true_values_path = sys.argv[1]
    predictions_path = sys.argv[2]
    error_threshold = float(sys.argv[3])
    metric = sys.argv[4]
    target_col = sys.argv[5]

    compare_predictions(
        true_values_path, predictions_path, error_threshold, metric, target_col
    )
