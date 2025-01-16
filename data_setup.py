# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import TfidfVectorizer

def scale_dataset(dataframe, oversample=False):
    """
    Scales the dataset using StandardScaler and optionally applies oversampling.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.
        oversample (bool): Whether to apply Random Oversampling. Defaults to False.

    Returns:
        tuple: Scaled data, feature matrix, and labels.
    """
    X = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)

    data = np.hstack((X, np.reshape(y, (-1, 1))))

    return data, X, y

def prepare_data(df):
    """
    Prepares the dataset by cleaning and splitting it into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: Training and testing sets for features and labels.
    """
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True, axis=1)

    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    
    df.drop_duplicates(inplace=True)

    df['label'] = (df['label'] == 'spam').astype(int)
    
    X = df['message']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
    tfidf_vectorizer = TfidfVectorizer()

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer

def prepare_single_message(message: str, vectorizer: TfidfVectorizer):
    """
    Prepares a single message for prediction.

    Args:
        message (str): The input message to prepare.
        vectorizer (TfidfVectorizer): The fitted TfidfVectorizer used for transforming the message.

    Returns:
        scipy.sparse matrix: The transformed message ready for prediction.
    """
    
    message_tfidf = vectorizer.transform([message])
    return message_tfidf