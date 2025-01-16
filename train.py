# train.py

from data_setup import prepare_data
from utils import *
from sklearn.svm import SVC

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./data/spam.csv', encoding='ISO-8859-1')

    X_train, X_test, y_train, y_test, tfidf_vectorizer = prepare_data(df)

    # Train SVM model
    svc_classifier = SVC(kernel='linear')
    svc_classifier.fit(X_train, y_train) 
    # Make predictions
    y_pred_svc = svc_classifier.predict(X_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred_svc)

    # Save the trained model
    save_model(svc_classifier, 'spam_classifier')
    save_model(tfidf_vectorizer, 'tfidf_vectorizer')