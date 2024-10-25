from flask import Flask, render_template, request, jsonify, Response
import json
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import spacy

app = Flask(__name__)

# Load the SpaCy model for text preprocessing
nlp = spacy.load('en_core_web_sm')

# Load pre-trained models and tokenizer
nlp_model = load_model("F:\\Project\\MentalHealthChatbot\\MentalHealthChatbot\\models\\nlp_model.h5")
cv_model = load_model("F:\\Project\\MentalHealthChatbot\\MentalHealthChatbot\\models\\cv_model.h5")
 # For facial expression recognition
with open(r"F:\Project\MentalHealthChatbot\MentalHealthChatbot\models\tokenizer.json") as f:

    data = json.load(f)
    tokenizer_json_string = json.dumps(data)
    tokenizer = tokenizer_from_json(tokenizer_json_string)

# Max sequence length for text input (should match the one used during training)
max_sequence_length = 100

# Function to process user text input for sentiment analysis
def process_text_input(text):
    doc = nlp(text)
    tokens = [token.text for token in doc]
    sequences = tokenizer.texts_to_sequences([tokens])
    padded_sequence = pad_sequences(sequences, maxlen=max_sequence_length)
    return padded_sequence

# Route to render the homepage (index.html)
@app.route('/')
def home():
    return render_template('index.html')

# Route to ask a random question
@app.route('/ask_question', methods=['GET'])
def ask_question():
    questions = [
        "How are you feeling today?",
        "Can you tell me about your day?",
        "What made you smile recently?",
        "Are you feeling anxious about anything?",
        "What are you looking forward to?"
    ]
    question = np.random.choice(questions)
    return jsonify({'question': question})

# Video feed route to stream the webcam for facial expression analysis
@app.route('/video_feed')
def video_feed():
    def generate():
        cap = cv2.VideoCapture(0)  # Open the webcam
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess the frame for facial expression prediction
            face = cv2.resize(frame, (48, 48))  # Resize to match model input
            face = face.astype('float32') / 255.0  # Normalize pixel values
            face = np.expand_dims(face, axis=0)

            # Predict emotion
            emotion_prediction = cv_model.predict(face)
            emotion_label = np.argmax(emotion_prediction)  # Assuming the output is one-hot encoded
            emotion_map = {0: 'Happy', 1: 'Sad', 2: 'Anxious'}  # Example map
            emotion = emotion_map.get(emotion_label, 'Neutral')

            # Encode the frame in JPEG format
            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            # Return the frame with emotion label overlaid
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to submit user's answer for sentiment prediction
@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    user_input = request.json.get('text', '')

    # Process text for sentiment prediction
    processed_text = process_text_input(user_input)
    text_sentiment = nlp_model.predict(processed_text)
    sentiment_label = np.argmax(text_sentiment)

    # Convert sentiment label to human-readable form
    sentiment_map = {0: 'Happy', 1: 'Sad', 2: 'Anxious'}
    predicted_sentiment = sentiment_map.get(sentiment_label, 'Neutral')

    return jsonify({'sentiment': predicted_sentiment})

# Main application entry point
if __name__ == '__main__':
    app.run(debug=True)
