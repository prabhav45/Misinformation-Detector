# Misinformation Detection System

A web application that uses machine learning to detect fake news articles. The system uses a PassiveAggressive Classifier trained on news articles to classify text as either FAKE or REAL with 92.74% accuracy.

## Features

- 🤖 AI-powered fake news detection
- 🎨 Modern, responsive web interface
- 📊 Confidence scores for predictions
- ⚡ Fast real-time analysis

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train and Save the Model

1. Open `Project 1 Misinformation.ipynb` in Jupyter Lab or VS Code
2. Run all cells in the notebook (this will train the model)
3. The last cell will save the model files:
   - `tfidf_vectorizer.pkl` - The text vectorizer
   - `pac_model.pkl` - The trained PassiveAggressive classifier

**Important:** Make sure you have the `news.zip` file in the same directory as the notebook.

### 3. Run the Web Application

```bash
python app.py
```

The application will start on `http://localhost:5000`

### 4. Use the Application

1. Open your web browser and navigate to `http://localhost:5000`
2. Paste a news article or text into the text area
3. Click "Analyze Article" to get the prediction
4. View the results showing whether the text is classified as FAKE or REAL, along with confidence scores

## Project Structure

```
ML Projects/
├── Project 1 Misinformation.ipynb  # Jupyter notebook with model training
├── app.py                          # Flask web application
├── templates/
│   └── index.html                  # Frontend HTML template
├── requirements.txt                # Python dependencies
├── tfidf_vectorizer.pkl            # Saved vectorizer (created after training)
├── pac_model.pkl                   # Saved model (created after training)
└── README.md                       # This file
```

## Model Details

- **Algorithm:** PassiveAggressive Classifier
- **Accuracy:** 92.74%
- **Features:** TF-IDF vectorization with English stop words
- **Training:** 80% train, 20% test split

## API Endpoints

### POST /predict
Analyze text and return prediction.

**Request:**
```json
{
  "text": "Your news article text here..."
}
```

**Response:**
```json
{
  "prediction": "FAKE",
  "confidence": {
    "FAKE": 0.85,
    "REAL": 0.15
  }
}
```

### GET /health
Check if the model is loaded and ready.

## Troubleshooting

- **Model not found error:** Make sure you've run the notebook and saved the model files
- **Port already in use:** Change the port in `app.py` (line 50) from `5000` to another port
- **Import errors:** Make sure all dependencies are installed: `pip install -r requirements.txt`

## Notes

- The model requires at least 10 characters of text to analyze
- Results are based on the training data and may not be 100% accurate
- Always verify important information through multiple reliable sources

