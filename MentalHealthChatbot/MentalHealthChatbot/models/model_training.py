import os
import sys

# Ensure the project root is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importing the necessary modules from models
from models import nlp_model, lstm_model, cv_model

def train_nlp_model():
    print("Starting training for NLP model...")
    nlp_model.train()
    print("NLP model training complete and saved successfully.\n")

def train_lstm_model():
    print("Starting training for LSTM model...")
    lstm_model.train()
    print("LSTM model training complete and saved successfully.\n")

def train_cv_model():
    print("Starting training for Computer Vision model...")
    cv_model.train()
    print("Computer Vision model training complete and saved successfully.\n")

def main():
    # Check and create necessary directories
    model_dirs = ['models']
    for model_dir in model_dirs:
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    print("Training all models...")
    train_nlp_model()
    train_lstm_model()
    train_cv_model()
    print("All models trained successfully.")

if __name__ == "__main__":
    main()
