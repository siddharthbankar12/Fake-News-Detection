from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import string

app = Flask(__name__)

# Load the models
models = []
for i in range(4):
    with open(f"models/model_{i}.pkl", 'rb') as file:
        model = pickle.load(file)
        models.append(model)

# Load the vectorizer
with open("models/vectorizer.pkl", 'rb') as file:
    vectorizer = pickle.load(file)

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        processed_text = wordopt(text)
        text_vectorized = vectorizer.transform([processed_text])
        predictions = [model.predict(text_vectorized)[0] for model in models]
        prediction_labels = ["Fake News" if pred == 0 else "True News" for pred in predictions]
        return render_template('index.html', prediction=prediction_labels, text=text)

if __name__ == '__main__':
    app.run(debug=True)
