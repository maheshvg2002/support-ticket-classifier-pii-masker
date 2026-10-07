import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

MODEL_PATH = "email_classifier_pipeline.pkl"

def train_and_save_model(csv_path="combined_emails_with_natural_pii.csv"):
    """
    Loads dataset, trains a TF-IDF + Logistic Regression model, and saves it.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['email', 'type'])
    
    X = df['email']
    y = df['type']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, C=1.0))
    ])
    
    pipeline.fit(X_train, y_train)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline

def load_model():
    """
    Loads the pre-trained classification pipeline or trains one if missing.
    """
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return train_and_save_model()