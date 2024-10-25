import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load raw data (Placeholder)
def load_raw_data():
    # Replace this with code to load your actual raw data
    data = {
        'text': [
            "I'm feeling great today!", "I don't know what to do anymore.",
            "This is so exciting!", "I'm really worried about my exam.",
            "Everything is falling apart.", "I'm looking forward to the weekend.",
            "Why is everything so difficult?", "I'm anxious about meeting new people.",
            "Life is beautiful.", "I can't handle this stress."
        ],
        'label': [
            'happy', 'sad', 'happy', 'anxious',
            'sad', 'happy', 'sad', 'anxious',
            'happy', 'anxious'
        ]
    }
    return pd.DataFrame(data)

# Function to preprocess text data
def preprocess_text(df):
    # Example preprocessing (lowercasing)
    df['text'] = df['text'].str.lower()
    return df

# Function to split and save dataset
def split_and_save_data(df):
    # Split dataset into training and testing sets
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    
    # Create directories if they don't exist
    processed_data_dir = 'data/processed'
    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)
    
    # Save the datasets
    train.to_csv(os.path.join(processed_data_dir, 'mental_health_train.csv'), index=False)
    test.to_csv(os.path.join(processed_data_dir, 'mental_health_test.csv'), index=False)
    print("Datasets saved successfully.")

def main():
    # Load raw data
    raw_data = load_raw_data()
    
    # Preprocess data
    processed_data = preprocess_text(raw_data)
    
    # Split and save data
    split_and_save_data(processed_data)

if __name__ == "__main__":
    main()
