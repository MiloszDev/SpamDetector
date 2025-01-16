from joblib import load
from pathlib import Path
from data_setup import prepare_single_message

# Define the model directory
model_dir = Path('models/')

def predict(message: str, model_path: str, vectorizer_path: str):
    """
    Predicts whether a given message is spam or ham.

    Args:
        message (str): The message to be classified.
        model_path (str): The path to the saved model file.
        vectorizer_path (str): The path to the saved TfidfVectorizer file.

    Returns:
        str: 'Spam' or 'Ham' based on the prediction.
    """
    model = load(model_path)

    vectorizer = load(vectorizer_path)

    prepared_message = prepare_single_message(message, vectorizer)

    prediction = model.predict(prepared_message)

    return 'Spam' if prediction[0] == 1 else 'Not Spam'

if __name__ == '__main__':
    message_to_predict = "You won a free iphone 13. Please give me your bank account number"
    model_file_path = model_dir / 'spam_classifier.joblib'
    vectorizer_file_path = model_dir / 'tfidf_vectorizer.joblib'

    result = predict(message_to_predict, model_file_path, vectorizer_file_path)
    print(f"The message is classified as: {result}")
