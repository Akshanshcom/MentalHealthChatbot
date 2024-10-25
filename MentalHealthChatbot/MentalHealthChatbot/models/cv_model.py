import cv2
import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Function to load and preprocess images
def load_and_preprocess_images(image_paths, labels, label_to_int):
    images = []
    integer_labels = []
    for img_path, label in zip(image_paths, labels):
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (48, 48))
                images.append(img)
                integer_labels.append(label_to_int[label])
            else:
                print(f"Error: Image at {img_path} could not be loaded.")
        else:
            print(f"Error: Image path {img_path} does not exist.")
    if not images:
        raise ValueError("No valid images found.")
    images = np.array(images)
    images = images / 255.0  # Normalize pixel values
    labels = to_categorical(integer_labels, num_classes=len(label_to_int))
    return images, labels

def train():
    # Load the dataset (assuming you have a CSV file with image paths and labels)
    df = pd.read_csv('data/processed/facial_expressions.csv')
    
    # Map labels to integers
    label_to_int = {label: idx for idx, label in enumerate(df['label'].unique())}
    
    # Preprocess the images and labels
    X, y = load_and_preprocess_images(df['image_path'].values, df['label'].values, label_to_int)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Reshape the data to fit the model
    X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)
    X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)
    
    # Build the CNN model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.2, verbose=2)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    
    # Save the model
    model.save('models/cv_model.h5')
    
    print("CNN Model training complete and saved successfully.")

# Allow this script to be run directly
if __name__ == "__main__":
    train()
