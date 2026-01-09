from flask import Flask, render_template, request, jsonify
import pickle
import os
import warnings
import numpy as np
from scipy.special import expit
warnings.filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)

# Load model and vectorizer
def load_model():
    try:
        with open('tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('pac_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        return vectorizer, model
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please make sure to run the notebook cell that saves the model first.")
        return None, None

vectorizer, model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Please train and save the model first.'}), 500
    
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text is too short. Please provide at least 10 characters.'}), 400
        
        # Transform and predict
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)[0]
        
        # Get decision function scores (PassiveAggressiveClassifier doesn't have predict_proba)
        decision_scores = model.decision_function(text_vectorized)[0]
        
        # Convert decision scores to probabilities using sigmoid
        # Get class labels in the correct order
        classes = model.classes_
        proba_dict = {}
        
        # For binary classification, decision_function returns a single value
        # Positive values favor one class, negative values favor the other
        if len(classes) == 2:
            # Convert to probability using sigmoid
            prob_positive = expit(decision_scores)
            prob_negative = 1 - prob_positive
            
            # Map to class labels (assuming classes are ['FAKE', 'REAL'])
            if classes[0] == 'FAKE':
                proba_dict['FAKE'] = float(prob_negative)
                proba_dict['REAL'] = float(prob_positive)
            else:
                proba_dict['FAKE'] = float(prob_positive)
                proba_dict['REAL'] = float(prob_negative)
        else:
            # For multi-class, use softmax
            exp_scores = np.exp(decision_scores - np.max(decision_scores))
            probabilities = exp_scores / np.sum(exp_scores)
            for i, cls in enumerate(classes):
                proba_dict[cls] = float(probabilities[i])
        
        return jsonify({
            'prediction': prediction,
            'confidence': proba_dict
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    if model is None or vectorizer is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'ok', 'message': 'Model loaded successfully'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

