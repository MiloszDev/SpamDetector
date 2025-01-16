# Spam Detector

This project implements a spam detection system using **Scikit-Learn** with an SVM model, achieving high accuracy in classifying messages as spam or not spam. The system is trained on a labeled SMS dataset and leverages **TF-IDF vectorization** for text preprocessing.

---

## Features

- **Text Classification**: Classifies messages as 'Spam' or 'Not Spam.'
- **TF-IDF Vectorization**: Converts text into numerical features.
- **SVM Model**: Uses a linear kernel for efficient classification.
- **Model Persistence**: Save and load models using `joblib`.

---

## Setup and Usage

1. **Clone the repository**:
    
    ```bash
    git clone https://github.com/yourusername/spam-detection.git
    cd spam-detection
    ```
    
2. **Install dependencies**:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3. **Train the model**:
    
    ```bash
    python train.py
    ```
    
4. **Predict a message**:
Update the `message_to_predict` in `predict.py` and run:
    
    ```bash
    python predict.py
    ```
    

---

## Files Overview

- `train.py`: Script for training the SVM model and saving it.
- `predict.py`: Script for loading the trained model and making predictions.
- `data_setup.py`: Functions for data preprocessing and TF-IDF vectorization.
- `utils.py`: Includes functions for saving models and evaluating performance.

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1 Score
- Confusion Matrix visualization.

---

## Data

- **Source**: SMS spam dataset (e.g., from UCI repository).
- **Processing**: Removes duplicates and irrelevant columns, and labels data as 1 (spam) or 0 (not spam).

---

## License

This project is licensed under the MIT License.
