import unittest
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import sys

# Ensure the path is correctly set
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the image preprocessing functions from cv_utils
from utils import cv_utils

# Define the test class
class TestCVModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the trained model
        cls.model = load_model('models/cv_model.h5')

        # Define the image size
        cls.img_size = (48, 48)

        # Define label mapping
        cls.label_to_int = {'happy': 0, 'sad': 1, 'angry': 2, 'surprised': 3, 'neutral': 4}
        cls.int_to_label = {v: k for k, v in cls.label_to_int.items()}

    def test_preprocess_image(self):
        # Test the image preprocessing function
        sample_image_path = 'data/processed/images\surprised\surprised_9.jpg'  # Ensure this path points to a valid image file
        if not os.path.exists(sample_image_path):
            self.fail(f"Test image not found: {sample_image_path}")
        preprocessed_img = cv_utils.preprocess_image(sample_image_path, size=self.img_size)
        self.assertEqual(preprocessed_img.shape, self.img_size)

    def test_model_prediction(self):
        # Test the model's prediction
        sample_image_path = 'data/processed/images\sad\sad_38.jpg'  # Ensure this path points to a valid image file
        if not os.path.exists(sample_image_path):
            self.fail(f"Test image not found: {sample_image_path}")
        preprocessed_img = cv_utils.preprocess_image(sample_image_path, size=self.img_size)
        preprocessed_img = np.reshape(preprocessed_img, (1, 48, 48, 1))

        # Predict the expression
        predictions = self.model.predict(preprocessed_img)
        predicted_label = np.argmax(predictions, axis=1)[0]

        self.assertIn(predicted_label, self.int_to_label.keys())

# Run the tests
if __name__ == '__main__':
    unittest.main()
