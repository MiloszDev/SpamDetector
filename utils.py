from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import joblib

def save_model(model, model_name, directory='models/'):
    """
    Saves a trained model to a specified directory.

    Args:
        model: The trained model to save.
        model_name (str): The name to use for the saved model file (without extension).
        directory (str): The directory where the model will be saved. Defaults to 'models/'.
    """
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # Construct the full path for the model file
    model_path = os.path.join(directory, f"{model_name}.joblib")

    # Save the model
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
def evaluate_model(y_true, y_pred):
    """
    Evaluates the performance of a classification model.

    Args:
        y_true (array-like): True labels of the dataset.
        y_pred (array-like): Predicted labels by the model.

    Returns:
        dict: A dictionary containing accuracy, precision, recall, F1 score, and confusion matrix.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': conf_matrix
    }
